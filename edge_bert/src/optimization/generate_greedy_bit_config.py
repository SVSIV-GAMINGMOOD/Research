import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import GREEDY_SETTINGS, MODELS_DIR, load_sensitivity_results


def main() -> None:
    sensitivity_results = load_sensitivity_results()
    if not sensitivity_results:
        raise FileNotFoundError("Run analyze_layer_sensitivity.py first to generate sensitivity_results.json.")

    sensitivity = sensitivity_results["sensitivity_by_layer"]
    locked_layers = set(sensitivity_results["locked_layers"])
    search_layers = {layer: drop for layer, drop in sensitivity.items() if layer not in locked_layers}
    sorted_layers = sorted(search_layers.items(), key=lambda item: item[1], reverse=True)

    if GREEDY_SETTINGS["allocation_strategy"] != "tertiles":
        raise ValueError(f"Unsupported greedy allocation strategy: {GREEDY_SETTINGS['allocation_strategy']}")

    count = len(sorted_layers)
    top = sorted_layers[: count // 3]
    mid = sorted_layers[count // 3 : 2 * count // 3]
    low = sorted_layers[2 * count // 3 :]

    greedy_config = {}
    for layer, _ in top:
        greedy_config[layer] = 8
    for layer, _ in mid:
        greedy_config[layer] = 6
    for layer, _ in low:
        greedy_config[layer] = 4
    for layer in locked_layers:
        greedy_config[layer] = 8

    print("\nGreedy Bit Allocation\n")
    for layer, bits in greedy_config.items():
        print(layer, "->", bits)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "greedy_config.json", "w", encoding="utf-8") as file:
        json.dump(greedy_config, file, indent=4)

    print(f"\nSaved to {MODELS_DIR / 'greedy_config.json'}")


if __name__ == "__main__":
    main()
