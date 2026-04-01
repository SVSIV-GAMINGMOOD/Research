import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from quantization_energy_function import (
    compute_energy,
    base_model,
    LOCKED_LAYERS,
    BIT_OPTIONS
)

# ==========================
# Build Searchable Layer List
# ==========================

searchable_layers = []

for name, module in base_model.named_modules():
    if hasattr(module, "weight"):
        if "Linear" in str(type(module)):
            if name not in LOCKED_LAYERS:
                searchable_layers.append(name)

print(f"Searchable Layers: {len(searchable_layers)}")

# ==========================
# Candidate 1: All 8-bit
# ==========================

candidate_all8 = {layer: 8 for layer in searchable_layers}

energy, acc, size, latency = compute_energy(candidate_all8)

print("\n===== ALL 8-BIT =====")
print(f"Accuracy: {acc:.4f}")
print(f"Estimated Size: {size:.2f} MB")
print(f"Estimated Latency: {latency:.2f} ms")
print(f"Energy: {energy:.4f}")

# ==========================
# Candidate 2: All 4-bit
# ==========================

candidate_all4 = {layer: 4 for layer in searchable_layers}

energy, acc, size, latency = compute_energy(candidate_all4)

print("\n===== ALL 4-BIT =====")
print(f"Accuracy: {acc:.4f}")
print(f"Estimated Size: {size:.2f} MB")
print(f"Estimated Latency: {latency:.2f} ms")
print(f"Energy: {energy:.4f}")

# ==========================
# Candidate 3: Random
# ==========================

candidate_random = {
    layer: random.choice(BIT_OPTIONS)
    for layer in searchable_layers
}

energy, acc, size, latency = compute_energy(candidate_random)

print("\n===== RANDOM MIXED =====")
print(f"Accuracy: {acc:.4f}")
print(f"Estimated Size: {size:.2f} MB")
print(f"Estimated Latency: {latency:.2f} ms")
print(f"Energy: {energy:.4f}")
