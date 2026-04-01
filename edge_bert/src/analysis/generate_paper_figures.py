import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import FIGURES_DIR, RESULTS_DIR, get_locked_layers, load_sa_best_config


METHOD_ORDER = [
    "FP32 Baseline",
    "INT8 Uniform",
    "FAR Frozen FP32",
    "FAR Frozen INT8",
    "Greedy Mixed",
    "SA Mixed (v1)",
    "Hybrid INT8+SA (Ours)",
]

COLORS = {
    "FP32 Baseline": "#607D8B",
    "INT8 Uniform": "#2196F3",
    "FAR Frozen FP32": "#4CAF50",
    "FAR Frozen INT8": "#009688",
    "Greedy Mixed": "#FF9800",
    "SA Mixed (v1)": "#9C27B0",
    "Hybrid INT8+SA (Ours)": "#F44336",
}

SHORT_LABELS = {
    "FP32 Baseline": "FP32",
    "INT8 Uniform": "INT8",
    "FAR Frozen FP32": "FAR\nFP32",
    "FAR Frozen INT8": "FAR\nINT8",
    "Greedy Mixed": "Greedy",
    "SA Mixed (v1)": "SA v1",
    "Hybrid INT8+SA (Ours)": "Hybrid\n(Ours)",
}


def get(result_map, method, key, default=None):
    entry = result_map.get(method)
    if entry is None:
        return default
    return entry.get(key, default)


def build_sa_matrix():
    sa_config = load_sa_best_config()
    locked_layers = set(get_locked_layers())
    op_patterns = [
        ("Q_lin", "attention.q_lin"),
        ("K_lin", "attention.k_lin"),
        ("V_lin", "attention.v_lin"),
        ("Out_lin", "attention.out_lin"),
        ("FFN1", "ffn.lin1"),
        ("FFN2", "ffn.lin2"),
    ]
    matrix = []
    locked_coords = []
    for layer_index in range(6):
        row = []
        for op_index, (_, pattern) in enumerate(op_patterns):
            layer_name = f"distilbert.transformer.layer.{layer_index}.{pattern}"
            row.append(sa_config.get(layer_name, 8))
            if layer_name in locked_layers:
                locked_coords.append((layer_index, op_index))
        matrix.append(row)
    return np.array(matrix), [label for label, _ in op_patterns], locked_coords


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eval_results = json.loads((RESULTS_DIR / "all_model_results.json").read_text(encoding="utf-8"))
    size_results = json.loads((RESULTS_DIR / "size_comparison.json").read_text(encoding="utf-8"))

    fp32_acc = get(eval_results, "FP32 Baseline", "accuracy", 0.9117)
    fp32_size = get(size_results, "FP32 Baseline", "theoretical_mb", 255.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    int8_size = get(size_results, "INT8 Uniform", "theoretical_mb", 63.75)
    ax.axvspan(0, int8_size + 5, alpha=0.06, color="#4CAF50", label=f"INT8-size target (<={int8_size:.0f} MB)")
    ax.axhline(fp32_acc, color="#607D8B", linestyle=":", linewidth=1.2, label=f"FP32 accuracy ({fp32_acc:.4f})")
    plotted_methods = [
        method for method in METHOD_ORDER
        if get(eval_results, method, "accuracy") is not None and get(size_results, method, "theoretical_mb") is not None
    ]
    for method in plotted_methods:
        acc = get(eval_results, method, "accuracy")
        size = get(size_results, method, "theoretical_mb")
        color = COLORS[method]
        is_ours = "Ours" in method
        ax.scatter(size, acc, s=220 if is_ours else 100, c=color, zorder=5 if is_ours else 3,
                   marker="*" if is_ours else "o", edgecolors="white", linewidths=1.5)
        ax.annotate(method, (size, acc), xytext=(5, 5), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold" if is_ours else "normal")
    ax.set_xlabel("Theoretical Effective Model Size (MB)")
    ax.set_ylabel("SST-2 Accuracy")
    ax.set_title("Accuracy vs Model Size Trade-off", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_tradeoff_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    matrix, op_labels, locked_coords = build_sa_matrix()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.get_cmap("RdYlGn", 3)
    im = ax.imshow(matrix, cmap=cmap, vmin=3, vmax=9, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            bits = matrix[row_index, col_index]
            color = "white" if bits <= 4 else "black"
            ax.text(col_index, row_index, f"{bits}b", ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.set_xticks(range(len(op_labels)))
    ax.set_yticks(range(6))
    ax.set_xticklabels(op_labels, fontsize=11)
    ax.set_yticklabels([f"Layer {i}" for i in range(6)], fontsize=11)
    ax.set_title("SA Bit Allocation Across Transformer Layers", fontweight="bold")
    fig.colorbar(im, ax=ax, ticks=[4, 6, 8]).set_label("Bit Width")
    for row_index, col_index in locked_coords:
        ax.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="#1565C0", lw=2.5, linestyle="--"))
    locked_patch = mpatches.Patch(edgecolor="#1565C0", facecolor="none", linestyle="--", linewidth=2.5,
                                  label="Sensitivity-locked")
    ax.legend(handles=[locked_patch], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_bit_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    methods_for_bar = [method for method in METHOD_ORDER if get(size_results, method, "theoretical_mb") is not None]
    sizes_theory = [get(size_results, method, "theoretical_mb") for method in methods_for_bar]
    ratios = [fp32_size / size for size in sizes_theory]
    short_labels = [SHORT_LABELS[method] for method in methods_for_bar]
    bar_colors = [COLORS[method] for method in methods_for_bar]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.bar(short_labels, sizes_theory, color=bar_colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax1.axhline(fp32_size, color="black", linestyle="--", linewidth=1.4)
    ax1.set_ylabel("Theoretical Effective Size (MB)")
    ax1.set_title("Model Size (All Methods)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.bar(short_labels, ratios, color=bar_colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax2.set_ylabel("Compression Ratio (x)")
    ax2.set_title("Compression vs FP32", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_compression_bars.png", dpi=200, bbox_inches="tight")
    plt.close()

    methods_lat = [method for method in METHOD_ORDER if get(eval_results, method, "latency_avg") is not None]
    latencies = [get(eval_results, method, "latency_avg") for method in methods_lat]
    lat_labels = [SHORT_LABELS[method] for method in methods_lat]
    lat_colors = [COLORS[method] for method in methods_lat]
    lat_p95 = [get(eval_results, method, "latency_p95", 0) for method in methods_lat]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(lat_labels, latencies, color=lat_colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.errorbar(range(len(methods_lat)), latencies, yerr=[np.zeros(len(methods_lat)), [p - a for p, a in zip(lat_p95, latencies)]],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(20.0, color="#F44336", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Avg Inference Latency (ms)")
    ax.set_title("CPU Inference Latency - All Models", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_latency_bars.png", dpi=200, bbox_inches="tight")
    plt.close()

    methods_acc = [method for method in METHOD_ORDER if get(eval_results, method, "accuracy") is not None and method != "FP32 Baseline"]
    acc_drops = [(fp32_acc - get(eval_results, method, "accuracy")) * 100 for method in methods_acc]
    acc_labels = [SHORT_LABELS[method] for method in methods_acc]
    acc_colors = [COLORS[method] for method in methods_acc]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(acc_labels, acc_drops, color=acc_colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, color="#FF9800", linestyle="--", linewidth=1.5)
    ax.axhline(2.0, color="#F44336", linestyle="--", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Accuracy Drop vs FP32 (%)")
    ax.set_title("Accuracy Degradation Relative to FP32 Baseline", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_accuracy_drop.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nAll 5 figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
