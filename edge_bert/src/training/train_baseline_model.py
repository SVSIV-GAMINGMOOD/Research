import random
from pathlib import Path
import sys

import numpy as np
import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, get_scheduler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DATASET_SETTINGS, DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

dataset = load_dataset(DATASET_SETTINGS["name"], DATASET_SETTINGS["config"])
tokenizer = DistilBertTokenizerFast.from_pretrained(DEFAULT_MODEL_NAME)


def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=DATASET_SETTINGS["max_length"],
    )


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=16)

model = DistilBertForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME, num_labels=DEFAULT_NUM_LABELS)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs * len(train_loader),
)

tracker = EmissionsTracker()
tracker.start()
best_accuracy = 0
patience = 2
early_stop_counter = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1: {f1:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "baseline_best.pt")
        print("Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

emissions = tracker.stop()
print(f"Total training emissions (kg CO2eq): {emissions}")
print("Training complete.")
