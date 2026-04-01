import copy
import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
from sklearn.metrics import accuracy_score

BIT_SIMULATION = 4  # simulate 4-bit worst-case
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# =========================
# Load Dataset
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

val_dataset = dataset["validation"]

# =========================
# Load Base Model
# =========================
base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
base_model.load_state_dict(torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu"))
base_model.eval()

# =========================
# Fake Quantization Function
# =========================
def fake_quantize_tensor(tensor, bits):
    qmin = 0.
    qmax = 2.**bits - 1.

    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)

    q_tensor = zero_point + tensor / (scale + 1e-8)
    q_tensor.clamp_(qmin, qmax).round_()

    dq_tensor = scale * (q_tensor - zero_point)
    return dq_tensor

# =========================
# Baseline Accuracy
# =========================
def evaluate(model):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for sample in val_dataset:
            input_ids = sample["input_ids"].unsqueeze(0)
            attention_mask = sample["attention_mask"].unsqueeze(0)
            labels = sample["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()

            predictions.append(pred)
            true_labels.append(labels.item())

    return accuracy_score(true_labels, predictions)

baseline_acc = evaluate(base_model)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# =========================
# Sensitivity Loop
# =========================
sensitivities = []

for name, module in base_model.named_modules():
    if isinstance(module, torch.nn.Linear):

        model_copy = copy.deepcopy(base_model)

        # Find same layer in copy
        target_layer = dict(model_copy.named_modules())[name]

        # Fake quantize weights
        target_layer.weight.data = fake_quantize_tensor(
            target_layer.weight.data,
            BIT_SIMULATION
        )

        acc = evaluate(model_copy)
        drop = baseline_acc - acc

        sensitivities.append((name, drop))
        print(f"{name} → Accuracy Drop: {drop:.4f}")

# Sort by sensitivity
sensitivities.sort(key=lambda x: x[1], reverse=True)

print("\nMost Sensitive Layers:")
for layer, drop in sensitivities[:10]:
    print(layer, drop)
