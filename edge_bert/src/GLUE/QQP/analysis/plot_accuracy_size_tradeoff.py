from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.QQP.task_config import FIGURES_DIR, RESULTS_DIR


METHOD_ORDER = [
    "FP32 Research Reference",
    "Primary Baseline (FP16)",
    "INT8 Uniform",
    "Greedy Mixed",
    "SA (v1)",
    "Hybrid SA (ours)",
]

DISPLAY_LABELS = {
    "Primary Baseline (FP16)": "FP16 Baseline",
    "Greedy Mixed": "Greedy Optimization",
}

POINT_COLORS = {
    "FP32 Research Reference": "#607D8B",
    "Primary Baseline (FP16)": "#3F51B5",
    "INT8 Uniform": "#2196F3",
    "Greedy Mixed": "#FF9800",
    "SA (v1)": "#9C27B0",
    "Hybrid SA (ours)": "#F44336",
}


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
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.scatter(
        [item[1] for item in points],
        [item[2] for item in points],
        s=100,
        c=[POINT_COLORS.get(item[0], "#455A64") for item in points],
        edgecolors="white",
        linewidths=1.3,
        zorder=3,
    )

    for method, size_mb, accuracy in points:
        display_label = DISPLAY_LABELS.get(method, method)
        ax.text(
            size_mb + 2,
            accuracy,
            display_label,
            fontsize=8.6,
            color="#1F2933",
            fontweight="bold" if method in DISPLAY_LABELS or "ours" in method.lower() else "normal",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
            },
            zorder=4,
        )

    ax.set_xlabel("Theoretical Model Size (MB)", color="#1F2933")
    ax.set_ylabel("Accuracy", color="#1F2933")
    ax.set_title("QQP Accuracy vs Theoretical Model Size", fontweight="bold", color="#1F2933")
    ax.tick_params(axis="both", colors="#1F2933")
    ax.grid(True, alpha=0.3, linestyle="--")

    output_path = FIGURES_DIR / "accuracy_vs_size_tradeoff.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
