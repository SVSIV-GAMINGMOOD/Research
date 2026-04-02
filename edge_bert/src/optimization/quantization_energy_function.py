import copy
from pathlib import Path
import sys

import torch
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_BIT_OPTIONS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    get_baseline_reference_metrics,
    get_energy_weights,
    get_locked_layers,
)
from shared.model_workflows import MODELS_DIR, load_sst2_validation_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BIT_OPTIONS = DEFAULT_BIT_OPTIONS
LOCKED_LAYERS = set(get_locked_layers())
ALPHA, BETA, GAMMA = get_energy_weights()
BASELINE_SIZE_MB, BASELINE_LATENCY = get_baseline_reference_metrics()

val_dataset = load_sst2_validation_dataset(format_type="torch")
base_model = DistilBertForSequenceClassification.from_pretrained(
    DEFAULT_MODEL_NAME,
    num_labels=DEFAULT_NUM_LABELS,
)
base_model.load_state_dict(torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu"))
base_model.to(DEVICE)
base_model.eval()


def get_searchable_layers() -> list[str]:
    searchable_layers: list[str] = []
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name not in LOCKED_LAYERS:
            searchable_layers.append(name)
    return searchable_layers


SEARCHABLE_LAYERS = get_searchable_layers()


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = 0.0
    qmax = 2.0**bits - 1.0
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    q_tensor = zero_point + tensor / (scale + 1e-8)
    q_tensor.clamp_(qmin, qmax).round_()
    return scale * (q_tensor - zero_point)


def evaluate(model) -> float:
    predictions = []
    true_labels = []
    with torch.no_grad():
        for sample in val_dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(torch.argmax(outputs.logits, dim=-1).item())
            true_labels.append(sample["labels"].item())
    return accuracy_score(true_labels, predictions)


def estimate_model_size(candidate_bits: dict[str, int]) -> float:
    total_bits = 0
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            bits = 8 if name in LOCKED_LAYERS else candidate_bits.get(name, 8)
            total_bits += module.weight.numel() * bits
    return (total_bits / 8) / (1024**2)


def estimate_latency(candidate_bits: dict[str, int]) -> float:
    weighted_sum = 0
    total_params = 0
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            params = module.weight.numel()
            bits = 8 if name in LOCKED_LAYERS else candidate_bits.get(name, 8)
            weighted_sum += params * bits
            total_params += params * 8
    return BASELINE_LATENCY * (weighted_sum / total_params)


def measure_baseline_accuracy() -> float:
    return evaluate(base_model)


def build_uniform_candidate(bits: int) -> dict[str, int]:
    return {layer: bits for layer in SEARCHABLE_LAYERS}


def evaluate_candidate(candidate_bits: dict[str, int]) -> tuple[float, float, float]:
    model_copy = copy.deepcopy(base_model)
    for name, module in model_copy.named_modules():
        if isinstance(module, torch.nn.Linear):
            bits = 8 if name in LOCKED_LAYERS else candidate_bits.get(name, 8)
            module.weight.data = fake_quantize_tensor(module.weight.data, bits)

    accuracy = evaluate(model_copy)
    size_mb = estimate_model_size(candidate_bits)
    latency_ms = estimate_latency(candidate_bits)
    return accuracy, size_mb, latency_ms


def get_size_latency_bounds() -> dict[str, float]:
    min_bits = min(BIT_OPTIONS)
    max_bits = max(BIT_OPTIONS)
    min_candidate = build_uniform_candidate(min_bits)
    max_candidate = build_uniform_candidate(max_bits)
    min_size = estimate_model_size(min_candidate)
    min_latency = estimate_latency(min_candidate)
    max_size = estimate_model_size(max_candidate)
    max_latency = estimate_latency(max_candidate)
    return {
        "size_min": min(min_size, max_size),
        "size_max": max(min_size, max_size),
        "latency_min": min(min_latency, max_latency),
        "latency_max": max(min_latency, max_latency),
    }


def compute_energy(candidate_bits: dict[str, int]):
    accuracy, size_mb, latency_ms = evaluate_candidate(candidate_bits)
    energy = (
        ALPHA * (1 - accuracy)
        + BETA * (latency_ms / BASELINE_LATENCY)
        + GAMMA * (size_mb / BASELINE_SIZE_MB)
    )
    return energy, accuracy, size_mb, latency_ms
