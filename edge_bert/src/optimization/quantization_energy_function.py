import copy
import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ==========================
# Device Configuration
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================
# Configuration
# ==========================

BIT_OPTIONS = [4, 6, 8]

LOCKED_LAYERS = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1"
}

ALPHA = 5.0
BETA = 2.0
GAMMA = 1.0

BASELINE_SIZE_MB = 255.55
BASELINE_LATENCY = 14.67

# ==========================
# Load Dataset
# ==========================

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

# ==========================
# Load Base Model (ON GPU)
# ==========================

base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

base_model.load_state_dict(
    torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu")
)

base_model.to(DEVICE)
base_model.eval()

# ==========================
# Fake Quantization
# ==========================

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

# ==========================
# Evaluation (CUDA)
# ==========================

def evaluate(model):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for sample in val_dataset:

            input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            labels = sample["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()

            predictions.append(pred)
            true_labels.append(labels.item())

    return accuracy_score(true_labels, predictions)

# ==========================
# Size Estimation
# ==========================

def estimate_model_size(candidate_bits):
    total_bits = 0

    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_params = module.weight.numel()

            if name in LOCKED_LAYERS:
                bits = 8
            else:
                bits = candidate_bits[name]

            total_bits += weight_params * bits

    total_bytes = total_bits / 8
    return total_bytes / (1024 ** 2)

# ==========================
# Latency Estimation (Analytical)
# ==========================

def estimate_latency(candidate_bits):
    weighted_sum = 0
    total_params = 0

    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            params = module.weight.numel()

            if name in LOCKED_LAYERS:
                bits = 8
            else:
                bits = candidate_bits[name]

            weighted_sum += params * bits
            total_params += params * 8

    ratio = weighted_sum / total_params
    return BASELINE_LATENCY * ratio

# ==========================
# Energy Function
# ==========================

def compute_energy(candidate_bits):

    model_copy = copy.deepcopy(base_model)

    for name, module in model_copy.named_modules():
        if isinstance(module, torch.nn.Linear):

            if name in LOCKED_LAYERS:
                bits = 8
            else:
                bits = candidate_bits[name]

            module.weight.data = fake_quantize_tensor(
                module.weight.data,
                bits
            )

    acc = evaluate(model_copy)
    size = estimate_model_size(candidate_bits)
    latency = estimate_latency(candidate_bits)

    energy = (
        ALPHA * (1 - acc)
        + BETA * (latency / BASELINE_LATENCY)
        + GAMMA * (size / BASELINE_SIZE_MB)
    )

    return energy, acc, size, latency
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
