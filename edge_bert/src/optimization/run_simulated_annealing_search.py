import copy
import math
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from optimization.quantization_energy_function import BIT_OPTIONS, LOCKED_LAYERS, base_model, compute_energy
from shared.experiment_settings import FIGURES_DIR, SEARCH_SETTINGS, save_sa_search_results


INITIAL_TEMPERATURE = SEARCH_SETTINGS["initial_temperature"]
FINAL_TEMPERATURE = SEARCH_SETTINGS["final_temperature"]
COOLING_RATE = SEARCH_SETTINGS["cooling_rate"]
ITERATIONS_PER_TEMP = SEARCH_SETTINGS["iterations_per_temp"]
EARLY_STOP_PATIENCE = SEARCH_SETTINGS["early_stop_patience"]
PLATEAU_EPS = SEARCH_SETTINGS["plateau_eps"]
NUM_RUNS = SEARCH_SETTINGS["num_runs"]


searchable_layers = []
for name, module in base_model.named_modules():
    if hasattr(module, "weight") and "Linear" in str(type(module)) and name not in LOCKED_LAYERS:
        searchable_layers.append(name)


def run_simulated_annealing(seed: int) -> dict:
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
            possible_bits = [bit for bit in BIT_OPTIONS if bit != current_bit]
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
        "history": best_energy_history,
    }


def main() -> None:
    all_results = [run_simulated_annealing(seed) for seed in range(NUM_RUNS)]
    best_overall = min(all_results, key=lambda item: item["energy"])

    print("\n==============================")
    print("FINAL COMPARISON")
    print("==============================")
    for result in all_results:
        print(
            f"Seed {result['seed']} -> Energy {result['energy']:.4f} | "
            f"Acc {result['accuracy']:.4f} | Size {result['size']:.2f} | Lat {result['latency']:.2f}"
        )

    print("\nBest Overall Seed:", best_overall["seed"])
    print(f"Best Energy: {best_overall['energy']:.4f}")
    print(f"Best Accuracy: {best_overall['accuracy']:.4f}")
    print(f"Best Size: {best_overall['size']:.2f} MB")
    print(f"Best Latency: {best_overall['latency']:.2f} ms")

    save_sa_search_results(best_overall, all_results)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for result in all_results:
        plt.plot(result["history"], label=f"Seed {result['seed']}")
    plt.xlabel("Temperature Step")
    plt.ylabel("Best Energy")
    plt.title("SA Convergence (Multiple Seeds)")
    plt.legend()
    plt.savefig(FIGURES_DIR / "sa_search_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
