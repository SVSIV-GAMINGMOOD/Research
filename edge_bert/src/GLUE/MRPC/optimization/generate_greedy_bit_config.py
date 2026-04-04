from __future__ import annotations

from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.task_config import GREEDY_SETTINGS, greedy_config_path, load_sensitivity_results, save_json


def main() -> None:
    sensitivity_results = load_sensitivity_results()
    if not sensitivity_results:
        raise FileNotFoundError("Run MRPC analyze_layer_sensitivity.py first to generate sensitivity results.")

    sensitivity = sensitivity_results["sensitivity_by_layer"]
    locked_layers = set(sensitivity_results["locked_layers"])
    search_layers = {layer: drop for layer, drop in sensitivity.items() if layer not in locked_layers}
    sorted_layers = sorted(search_layers.items(), key=lambda item: item[1], reverse=True)

    if GREEDY_SETTINGS["allocation_strategy"] != "tertiles":
        raise ValueError(f"Unsupported MRPC greedy allocation strategy: {GREEDY_SETTINGS['allocation_strategy']}")

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

    save_json(greedy_config_path(), greedy_config)
    print(f"MRPC greedy bit allocation saved to {greedy_config_path()}")


if __name__ == "__main__":
    main()

