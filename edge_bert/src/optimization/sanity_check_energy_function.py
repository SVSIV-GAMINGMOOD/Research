import random
from pathlib import Path
import sys

import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

from quantization_energy_function import (
    BIT_OPTIONS,
    LOCKED_LAYERS,
    SEARCHABLE_LAYERS,
    base_model,
    compute_energy,
    measure_baseline_accuracy,
)


random.seed(42)


def print_candidate_summary(title: str, energy: float, acc: float, size: float, latency: float, threshold: float, baseline_acc: float) -> None:
    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Estimated Size: {size:.2f} MB")
    print(f"Estimated Latency: {latency:.2f} ms")
    print(f"Energy: {energy:.4f}")
    print(f"Feasible: {acc >= threshold}")
    print(f"Accuracy Drop: {baseline_acc - acc:.4f}")


# ==========================
# Baseline + Searchable Layers
# ==========================

baseline_acc = measure_baseline_accuracy()
print(f"\nBaseline FP32 Accuracy: {baseline_acc:.4f}")

searchable_layers = list(SEARCHABLE_LAYERS)

if not searchable_layers:
    raise ValueError("No searchable layers found")

print(f"Searchable Layers: {len(searchable_layers)}")
print(f"Locked Layers: {len(LOCKED_LAYERS)}")

linear_layer_count = sum(1 for _, module in base_model.named_modules() if isinstance(module, nn.Linear))
print(f"Detected Linear Layers: {linear_layer_count}")

threshold = baseline_acc * 0.98
print(f"Feasibility Threshold (98% of baseline): {threshold:.4f}")


# ==========================
# Candidate 1: All max-bit
# ==========================

max_bit = max(BIT_OPTIONS)
candidate_all_max = {layer: max_bit for layer in searchable_layers}

energy, acc, size, latency = compute_energy(candidate_all_max)
print_candidate_summary(f"ALL {max_bit}-BIT", energy, acc, size, latency, threshold, baseline_acc)


# ==========================
# Candidate 2: All min-bit
# ==========================

min_bit = min(BIT_OPTIONS)
candidate_all_min = {layer: min_bit for layer in searchable_layers}

energy, acc, size, latency = compute_energy(candidate_all_min)
print_candidate_summary(f"ALL {min_bit}-BIT", energy, acc, size, latency, threshold, baseline_acc)


# ==========================
# Candidate 3: Random
# ==========================

print("\n===== RANDOM MIXED (MULTIPLE) =====")

for index in range(5):
    candidate_random = {
        layer: random.choice(BIT_OPTIONS)
        for layer in searchable_layers
    }

    energy, acc, size, latency = compute_energy(candidate_random)

    print(f"\n-- Sample {index + 1} --")
    print(f"Accuracy: {acc:.4f}")
    print(f"Estimated Size: {size:.2f} MB")
    print(f"Estimated Latency: {latency:.2f} ms")
    print(f"Energy: {energy:.4f}")
    print(f"Feasible: {acc >= threshold}")
    print(f"Accuracy Drop: {baseline_acc - acc:.4f}")
