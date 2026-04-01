import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
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
# SA Hyperparameters
# ==========================

INITIAL_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.05
COOLING_RATE = 0.90
ITERATIONS_PER_TEMP = 3

EARLY_STOP_PATIENCE = 8
PLATEAU_EPS = 1e-3

NUM_RUNS = 3   # <- Run 3 different seeds

# ==========================
# Build Searchable Layers
# ==========================

searchable_layers = []

for name, module in base_model.named_modules():
    if hasattr(module, "weight"):
        if "Linear" in str(type(module)):
            if name not in LOCKED_LAYERS:
                searchable_layers.append(name)

# ==========================
# SA Function
# ==========================

def run_simulated_annealing(seed):

    print(f"\n==============================")
    print(f"Running SA with seed {seed}")
    print(f"==============================")

    random.seed(seed)
    np.random.seed(seed)

    current_candidate = {layer: 8 for layer in searchable_layers}
    current_energy, current_acc, current_size, current_latency = compute_energy(current_candidate)

    best_candidate = copy.deepcopy(current_candidate)
    best_energy = current_energy
    best_acc = current_acc
    best_size = current_size
    best_latency = current_latency

    temperature = INITIAL_TEMPERATURE
    no_improve_counter = 0

    best_energy_history = []

    while temperature > FINAL_TEMPERATURE:

        temp_best_energy = best_energy

        for _ in range(ITERATIONS_PER_TEMP):

            new_candidate = copy.deepcopy(current_candidate)

            layer_to_modify = random.choice(searchable_layers)
            current_bit = new_candidate[layer_to_modify]

            possible_bits = [b for b in BIT_OPTIONS if b != current_bit]
            new_candidate[layer_to_modify] = random.choice(possible_bits)

            new_energy, new_acc, new_size, new_latency = compute_energy(new_candidate)

            delta = new_energy - current_energy

            if delta < 0 or random.random() < math.exp(-delta / temperature):

                current_candidate = new_candidate
                current_energy = new_energy
                current_acc = new_acc
                current_size = new_size
                current_latency = new_latency

                if current_energy < best_energy:
                    best_candidate = copy.deepcopy(current_candidate)
                    best_energy = current_energy
                    best_acc = current_acc
                    best_size = current_size
                    best_latency = current_latency

        best_energy_history.append(best_energy)

        improvement = temp_best_energy - best_energy

        if improvement < PLATEAU_EPS:
            no_improve_counter += 1
        else:
            no_improve_counter = 0

        if no_improve_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

        temperature *= COOLING_RATE

    print(f"\nSeed {seed} Results:")
    print(f"Energy: {best_energy:.4f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Size: {best_size:.2f} MB")
    print(f"Latency: {best_latency:.2f} ms")

    return {
        "seed": seed,
        "energy": best_energy,
        "accuracy": best_acc,
        "size": best_size,
        "latency": best_latency,
        "candidate": best_candidate,
        "history": best_energy_history
    }

# ==========================
# Run Multiple Seeds
# ==========================

all_results = []

for seed in range(NUM_RUNS):
    result = run_simulated_annealing(seed)
    all_results.append(result)

# ==========================
# Compare Results
# ==========================

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

best_overall = min(all_results, key=lambda x: x["energy"])

for r in all_results:
    print(f"Seed {r['seed']} → Energy {r['energy']:.4f} | Acc {r['accuracy']:.4f} | Size {r['size']:.2f} | Lat {r['latency']:.2f}")

print("\nBest Overall Seed:", best_overall["seed"])
print(f"Best Energy: {best_overall['energy']:.4f}")
print(f"Best Accuracy: {best_overall['accuracy']:.4f}")
print(f"Best Size: {best_overall['size']:.2f} MB")
print(f"Best Latency: {best_overall['latency']:.2f} ms")

# ==========================
# Plot Convergence Curves
# ==========================

plt.figure()
for r in all_results:
    plt.plot(r["history"], label=f"Seed {r['seed']}")

plt.xlabel("Temperature Step")
plt.ylabel("Best Energy")
plt.title("SA Convergence (Multiple Seeds)")
plt.legend()
plt.show()
