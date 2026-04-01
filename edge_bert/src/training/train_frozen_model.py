import os
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from codecarbon import EmissionsTracker

# =========================
# 1. Reproducibility
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# =========================
# 2. Load Dataset
# =========================
dataset = load_dataset("glue", "sst2")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=16)

# =========================
# 3. Model
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =========================
# 4. Freeze Early Layers
# =========================
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False

for layer in model.distilbert.transformer.layer[:2]:
    for param in layer.parameters():
        param.requires_grad = False

print("Frozen embeddings + first 2 transformer layers.")

model.to(device)

# =========================
# 5. Optimizer (Only Trainable Params)
# =========================
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# =========================
# 6. Training Loop
# =========================
tracker = EmissionsTracker()
tracker.start()

best_accuracy = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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
        torch.save(model.state_dict(), MODELS_DIR / "frozen_best.pt")
        print("Frozen model saved.")

emissions = tracker.stop()
print(f"Training emissions (kg CO2eq): {emissions}")
print("Frozen training complete.")
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
