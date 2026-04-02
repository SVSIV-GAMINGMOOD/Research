import math
import random
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from optimization.quantization_energy_function import (
    BIT_OPTIONS,
    SEARCHABLE_LAYERS,
    build_uniform_candidate,
    evaluate_candidate,
    get_size_latency_bounds,
    measure_baseline_accuracy,
)
from shared.experiment_settings import (
    FIGURES_DIR,
    SEARCH_SETTINGS,
    get_accuracy_thresholds,
    save_sa_search_results,
)


INITIAL_TEMPERATURE = SEARCH_SETTINGS["initial_temperature"]
FINAL_TEMPERATURE = SEARCH_SETTINGS["final_temperature"]
COOLING_RATE = SEARCH_SETTINGS["cooling_rate"]
ITERATIONS_PER_TEMP = SEARCH_SETTINGS["iterations_per_temp"]
EARLY_STOP_PATIENCE = SEARCH_SETTINGS["early_stop_patience"]
PLATEAU_EPS = SEARCH_SETTINGS["plateau_eps"]
NUM_RUNS = SEARCH_SETTINGS["num_runs"]
DEFAULT_THRESHOLD_INDEX = SEARCH_SETTINGS.get("default_threshold_index", 0)
GUIDE_LATENCY_WEIGHT = SEARCH_SETTINGS.get("guide_latency_weight", 1.0)
GUIDE_SIZE_WEIGHT = SEARCH_SETTINGS.get("guide_size_weight", 1.0)


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def is_feasible(accuracy: float, threshold_accuracy: float) -> bool:
    return accuracy >= threshold_accuracy


def dominates(candidate_a: dict, candidate_b: dict) -> bool:
    better_on_one = candidate_a["size"] < candidate_b["size"] or candidate_a["latency"] < candidate_b["latency"]
    no_worse = candidate_a["size"] <= candidate_b["size"] and candidate_a["latency"] <= candidate_b["latency"]
    return better_on_one and no_worse


def candidate_signature(candidate: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(candidate.items()))


def compute_guide_energy(size_mb: float, latency_ms: float, bounds: dict[str, float]) -> float:
    normalized_size = normalize(size_mb, bounds["size_min"], bounds["size_max"])
    normalized_latency = normalize(latency_ms, bounds["latency_min"], bounds["latency_max"])
    return (GUIDE_SIZE_WEIGHT * normalized_size) + (GUIDE_LATENCY_WEIGHT * normalized_latency)


def evaluate_candidate_record(
    candidate: dict[str, int],
    threshold_spec: dict[str, float | str],
    bounds: dict[str, float],
    seed: int,
) -> dict:
    accuracy, size_mb, latency_ms = evaluate_candidate(candidate)
    threshold_accuracy = float(threshold_spec["accuracy_threshold"])
    feasible = is_feasible(accuracy, threshold_accuracy)
    return {
        "seed": seed,
        "threshold_label": threshold_spec["label"],
        "threshold_drop_percent": threshold_spec["drop_percent"],
        "threshold_accuracy": threshold_accuracy,
        "accuracy": accuracy,
        "size": size_mb,
        "latency": latency_ms,
        "feasible": feasible,
        "guide_energy": compute_guide_energy(size_mb, latency_ms, bounds) if feasible else float("inf"),
        "candidate": deepcopy(candidate),
    }


def insert_frontier_candidate(frontier: list[dict], candidate_record: dict) -> list[dict]:
    if not candidate_record["feasible"]:
        return frontier

    signature = candidate_signature(candidate_record["candidate"])
    for existing in frontier:
        existing_signature = candidate_signature(existing["candidate"])
        if existing_signature == signature and existing["guide_energy"] <= candidate_record["guide_energy"]:
            return frontier
        if dominates(existing, candidate_record):
            return frontier

    pruned_frontier = [
        existing
        for existing in frontier
        if candidate_signature(existing["candidate"]) != signature and not dominates(candidate_record, existing)
    ]
    pruned_frontier.append(candidate_record)
    return sorted(pruned_frontier, key=lambda item: (item["size"], item["latency"], -item["accuracy"]))


def merge_frontiers(frontiers: list[list[dict]]) -> list[dict]:
    merged: list[dict] = []
    for frontier in frontiers:
        for candidate_record in frontier:
            merged = insert_frontier_candidate(merged, candidate_record)
    return merged


def run_simulated_annealing(seed: int, threshold_spec: dict[str, float | str], bounds: dict[str, float]) -> dict:
    print("\n==============================")
    print(f"Running constrained Pareto SA | seed={seed} | threshold={threshold_spec['accuracy_threshold']:.4f}")
    print("==============================")

    random.seed(seed)
    np.random.seed(seed)

    current_candidate = build_uniform_candidate(max(BIT_OPTIONS))
    current_record = evaluate_candidate_record(current_candidate, threshold_spec, bounds, seed)
    if not current_record["feasible"]:
        print("Skipping run because the all-8-bit starting point is already below the threshold.")
        return {
            "seed": seed,
            "threshold_label": threshold_spec["label"],
            "threshold_drop_percent": threshold_spec["drop_percent"],
            "threshold_accuracy": threshold_spec["accuracy_threshold"],
            "history": [],
            "best_result": None,
            "frontier": [],
            "feasible_start": False,
        }

    best_record = deepcopy(current_record)
    frontier = insert_frontier_candidate([], current_record)
    temperature = INITIAL_TEMPERATURE
    no_improve_counter = 0
    best_energy_history: list[float] = []

    while temperature > FINAL_TEMPERATURE:
        previous_best_energy = best_record["guide_energy"]
        for _ in range(ITERATIONS_PER_TEMP):
            new_candidate = dict(current_candidate)
            layer_to_modify = random.choice(SEARCHABLE_LAYERS)
            current_bit = new_candidate[layer_to_modify]
            possible_bits = [bit for bit in BIT_OPTIONS if bit != current_bit]
            new_candidate[layer_to_modify] = random.choice(possible_bits)

            new_record = evaluate_candidate_record(new_candidate, threshold_spec, bounds, seed)
            if not new_record["feasible"]:
                continue

            frontier = insert_frontier_candidate(frontier, new_record)
            delta = new_record["guide_energy"] - current_record["guide_energy"]

            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_candidate = dict(new_candidate)
                current_record = new_record

            if new_record["guide_energy"] < best_record["guide_energy"]:
                best_record = deepcopy(new_record)

        best_energy_history.append(best_record["guide_energy"])
        improvement = previous_best_energy - best_record["guide_energy"]
        if improvement < PLATEAU_EPS:
            no_improve_counter += 1
        else:
            no_improve_counter = 0
        if no_improve_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break
        temperature *= COOLING_RATE

    print(f"Seed {seed} best feasible point:")
    print(f"Accuracy: {best_record['accuracy']:.4f}")
    print(f"Size: {best_record['size']:.2f} MB")
    print(f"Latency: {best_record['latency']:.2f} ms")
    print(f"Guide Energy: {best_record['guide_energy']:.4f}")
    print(f"Frontier Points Found: {len(frontier)}")

    return {
        "seed": seed,
        "threshold_label": threshold_spec["label"],
        "threshold_drop_percent": threshold_spec["drop_percent"],
        "threshold_accuracy": threshold_spec["accuracy_threshold"],
        "history": best_energy_history,
        "best_result": best_record,
        "frontier": frontier,
        "feasible_start": True,
    }


def select_default_result(threshold_results: list[dict]) -> dict | None:
    if not threshold_results:
        return None

    feasible_thresholds = [result for result in threshold_results if result.get("representative_result")]
    if not feasible_thresholds:
        return None

    target_index = max(0, min(DEFAULT_THRESHOLD_INDEX, len(threshold_results) - 1))
    for index in range(target_index, len(threshold_results)):
        candidate = threshold_results[index].get("representative_result")
        if candidate:
            return candidate
    for index in range(target_index - 1, -1, -1):
        candidate = threshold_results[index].get("representative_result")
        if candidate:
            return candidate
    return feasible_thresholds[0]["representative_result"]


def plot_convergence(threshold_results: list[dict]) -> None:
    if not threshold_results:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(threshold_results), 1, figsize=(10, max(4.5, 3.2 * len(threshold_results))))
    if len(threshold_results) == 1:
        axes = [axes]

    for axis, threshold_result in zip(axes, threshold_results):
        for run in threshold_result["runs"]:
            if run["history"]:
                axis.plot(run["history"], label=f"Seed {run['seed']}")
        axis.set_title(
            f"{threshold_result['label']} | threshold={threshold_result['accuracy_threshold']:.4f} "
            f"({threshold_result['drop_percent']:.2f}% drop)"
        )
        axis.set_xlabel("Temperature Step")
        axis.set_ylabel("Best Guide Energy")
        axis.grid(True, alpha=0.3, linestyle="--")
        if any(run["history"] for run in threshold_result["runs"]):
            axis.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sa_search_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_pareto_frontiers(threshold_results: list[dict], baseline_accuracy: float) -> None:
    feasible_thresholds = [result for result in threshold_results if result["frontier"]]
    if not feasible_thresholds:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    color_map = plt.cm.get_cmap("viridis", len(feasible_thresholds))
    for index, threshold_result in enumerate(feasible_thresholds):
        frontier = threshold_result["frontier"]
        sizes = [point["size"] for point in frontier]
        accuracies = [point["accuracy"] for point in frontier]
        plt.plot(
            sizes,
            accuracies,
            marker="o",
            linewidth=1.5,
            color=color_map(index),
            label=f"-{threshold_result['drop_percent']:.2f}% threshold",
        )

    plt.axhline(baseline_accuracy, color="#607D8B", linestyle=":", linewidth=1.3, label="Baseline accuracy")
    plt.xlabel("Theoretical Model Size (MB)")
    plt.ylabel("Validation Accuracy")
    plt.title("Constrained Pareto Frontiers Across Accuracy Thresholds")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sa_pareto_frontiers.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    baseline_accuracy = measure_baseline_accuracy()
    threshold_specs = get_accuracy_thresholds(baseline_accuracy)
    bounds = get_size_latency_bounds()

    print("\n==============================")
    print("Constrained Pareto Search Setup")
    print("==============================")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Threshold drops (%): {[spec['drop_percent'] for spec in threshold_specs]}")
    print(f"Searchable layers: {len(SEARCHABLE_LAYERS)}")
    print(
        f"Size bounds: {bounds['size_min']:.2f} MB .. {bounds['size_max']:.2f} MB | "
        f"Latency bounds: {bounds['latency_min']:.2f} ms .. {bounds['latency_max']:.2f} ms"
    )

    threshold_results = []
    for threshold_spec in threshold_specs:
        runs = [run_simulated_annealing(seed, threshold_spec, bounds) for seed in range(NUM_RUNS)]
        merged_frontier = merge_frontiers([run["frontier"] for run in runs])
        representative_result = min(merged_frontier, key=lambda item: item["guide_energy"]) if merged_frontier else None

        threshold_result = {
            "label": threshold_spec["label"],
            "drop_percent": threshold_spec["drop_percent"],
            "accuracy_threshold": threshold_spec["accuracy_threshold"],
            "runs": runs,
            "frontier": merged_frontier,
            "representative_result": representative_result,
        }
        threshold_results.append(threshold_result)

        print("\n------------------------------")
        print(
            f"Threshold {threshold_spec['label']} -> frontier points: {len(merged_frontier)} "
            f"| feasible seeds: {sum(1 for run in runs if run['best_result'])}/{len(runs)}"
        )
        if representative_result:
            print(
                f"Representative point: acc={representative_result['accuracy']:.4f}, "
                f"size={representative_result['size']:.2f} MB, "
                f"latency={representative_result['latency']:.2f} ms"
            )

    default_result = select_default_result(threshold_results)
    search_payload = {
        "search_mode": "constrained_pareto",
        "baseline_accuracy": baseline_accuracy,
        "settings": {
            "search": SEARCH_SETTINGS,
            "bit_options": BIT_OPTIONS,
        },
        "normalization_bounds": bounds,
        "thresholds": threshold_results,
        "default_result": default_result,
    }
    save_sa_search_results(search_payload)

    if default_result:
        print("\n==============================")
        print("Default SA Config for Downstream Scripts")
        print("==============================")
        print(f"Threshold label: {default_result['threshold_label']}")
        print(f"Threshold accuracy: {default_result['threshold_accuracy']:.4f}")
        print(f"Accuracy: {default_result['accuracy']:.4f}")
        print(f"Size: {default_result['size']:.2f} MB")
        print(f"Latency: {default_result['latency']:.2f} ms")

    plot_convergence(threshold_results)
    plot_pareto_frontiers(threshold_results, baseline_accuracy)


if __name__ == "__main__":
    main()
