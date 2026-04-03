import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import FIGURES_DIR, RESULTS_DIR


METHOD_ORDER = [
    "FP32 Baseline",
    "INT8 Uniform",
    "FAR Frozen FP32",
    "FAR Frozen INT8",
    "Greedy Mixed",
    "SA Mixed (v1)",
    "Hybrid INT8+SA (Ours)",
]


def main() -> None:
    eval_results = json.loads((RESULTS_DIR / "all_model_results.json").read_text(encoding="utf-8"))
    size_results = json.loads((RESULTS_DIR / "size_comparison.json").read_text(encoding="utf-8"))

    points = []
    for method in METHOD_ORDER:
        eval_entry = eval_results.get(method)
        size_entry = size_results.get(method)
        if not eval_entry or not size_entry:
            continue
        if eval_entry.get("accuracy") is None or size_entry.get("theoretical_mb") is None:
            continue
        points.append((method, size_entry["theoretical_mb"], eval_entry["accuracy"]))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.scatter([item[1] for item in points], [item[2] for item in points], s=90)

    for method, size_mb, accuracy in points:
        plt.text(size_mb + 2, accuracy, method, fontsize=8.5)

    plt.xlabel("Theoretical Model Size (MB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Model Size Trade-off")
    plt.grid(True, alpha=0.3)

    output_path = FIGURES_DIR / "accuracy_vs_size_tradeoff.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
