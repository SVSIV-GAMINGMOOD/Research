"""
=============================================================
STAGE 6B — Paper Figures (ALL 7 Models)
=============================================================
Reads from:
  results/all_model_results.json   (from evaluate_all_onnx_models.py)
  results/size_comparison.json     (from compare_model_sizes.py)

Generates 5 publication-quality figures:
  fig1_tradeoff_curve.png      Accuracy vs Theoretical Size
  fig2_bit_heatmap.png         SA bit allocation across layers
  fig3_compression_bars.png    Size + ratio for all 7 methods
  fig4_latency_bars.png        Latency comparison all models
  fig5_accuracy_drop.png       Accuracy drop vs FP32 baseline

Run AFTER compare_model_sizes.py AND evaluate_all_onnx_models.py:
  python generate_paper_figures.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = SRC_ROOT / "results"
FIGURES_DIR = SRC_ROOT / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD RESULTS
# ----------------------------------------------------------
eval_path = RESULTS_DIR / "all_model_results.json"
size_path = RESULTS_DIR / "size_comparison.json"

with open(eval_path) as f:
    eval_results = json.load(f)

with open(size_path) as f:
    size_results = json.load(f)

# ----------------------------------------------------------
# UNIFIED DATA — merge eval + size per method
# ----------------------------------------------------------
# Display order for plots (progression of ideas)
METHOD_ORDER = [
    "FP32 Baseline",
    "INT8 Uniform",
    "FAR Frozen FP32",
    "FAR Frozen INT8",
    "Greedy Mixed",
    "SA Mixed (v1)",
    "Hybrid INT8+SA (Ours)",
]

# Color palette — distinct, colorblind-friendly
COLORS = {
    "FP32 Baseline":         "#607D8B",   # grey-blue
    "INT8 Uniform":          "#2196F3",   # blue
    "FAR Frozen FP32":       "#4CAF50",   # green
    "FAR Frozen INT8":       "#009688",   # teal
    "Greedy Mixed":          "#FF9800",   # orange
    "SA Mixed (v1)":         "#9C27B0",   # purple
    "Hybrid INT8+SA (Ours)": "#F44336",   # red  ← our method
}

# Short labels for tight plots
SHORT_LABELS = {
    "FP32 Baseline":         "FP32",
    "INT8 Uniform":          "INT8",
    "FAR Frozen FP32":       "FAR\nFP32",
    "FAR Frozen INT8":       "FAR\nINT8",
    "Greedy Mixed":          "Greedy",
    "SA Mixed (v1)":         "SA v1",
    "Hybrid INT8+SA (Ours)": "Hybrid\n(Ours)",
}

def get(method, key, default=None):
    """Safe accessor — returns default if method missing/None."""
    d = eval_results.get(method)
    if d is None:
        return default
    return d.get(key, default)

def get_size(method, key="theoretical_mb", default=None):
    d = size_results.get(method)
    if d is None:
        return default
    return d.get(key, default)

fp32_acc  = get("FP32 Baseline", "accuracy", 0.9117)
fp32_size = get_size("FP32 Baseline", "theoretical_mb", 255.0)

# ===========================================================
# FIGURE 1 — Accuracy vs Effective Model Size (Trade-off)
# ===========================================================
fig, ax = plt.subplots(figsize=(10, 6.5))

# Light target zone: INT8 size and better
int8_size = get_size("INT8 Uniform", "theoretical_mb", 63.75)
ax.axvspan(0, int8_size + 5, alpha=0.06, color="#4CAF50",
           label=f"INT8-size target (≤{int8_size:.0f} MB)")
ax.axhline(fp32_acc, color="#607D8B", linestyle=":", linewidth=1.2,
           label=f"FP32 accuracy ({fp32_acc:.4f})")

plotted_methods = [m for m in METHOD_ORDER
                   if get(m, "accuracy") is not None
                   and get_size(m) is not None]

for method in plotted_methods:
    acc   = get(method, "accuracy")
    size  = get_size(method)
    color = COLORS[method]
    is_ours = "Ours" in method

    ax.scatter(size, acc,
               s=220 if is_ours else 100,
               c=color,
               zorder=5 if is_ours else 3,
               marker="*" if is_ours else "o",
               edgecolors="white", linewidths=1.5)

    # Smart label offset to avoid overlap
    offsets = {
        "FP32 Baseline":         ( 5,  5),
        "INT8 Uniform":          (-45,-14),
        "FAR Frozen FP32":       ( 5, -13),
        "FAR Frozen INT8":       ( 5,   5),
        "Greedy Mixed":          ( 5,   5),
        "SA Mixed (v1)":         ( 5,   5),
        "Hybrid INT8+SA (Ours)": ( 5,   7),
    }
    dx, dy = offsets.get(method, (5, 5))
    ax.annotate(
        method,
        (size, acc),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=8.5,
        color=color,
        fontweight="bold" if is_ours else "normal",
    )

# Ideal direction arrow
ax.annotate("← Better\n(smaller + accurate)",
            xy=(55, ax.get_ylim()[0] + 0.003 if ax.get_ylim()[0] else 0.895),
            fontsize=8, color="gray", style="italic")

ax.set_xlabel("Theoretical Effective Model Size (MB)", fontsize=12)
ax.set_ylabel("SST-2 Accuracy", fontsize=12)
ax.set_title("Accuracy vs Model Size Trade-off\n"
             "FAR + Simulated Annealing Hybrid Quantization Framework",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, linestyle="--")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_tradeoff_curve.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig1_tradeoff_curve.png")

# ===========================================================
# FIGURE 2 — Bit Allocation Heatmap (SA config)
# ===========================================================
SA_MATRIX_CONFIG = {
    "L0": {"Q": 8,  "K": 6, "V": 4, "O": 8,  "FF1": 4, "FF2": 6},
    "L1": {"Q": 6,  "K": 4, "V": 4, "O": 6,  "FF1": 8, "FF2": 8},
    "L2": {"Q": 4,  "K": 6, "V": 4, "O": 6,  "FF1": 8, "FF2": 8},
    "L3": {"Q": 8,  "K": 6, "V": 4, "O": 8,  "FF1": 8, "FF2": 6},
    "L4": {"Q": 8,  "K": 8, "V": 8, "O": 4,  "FF1": 4, "FF2": 6},
    "L5": {"Q": 4,  "K": 4, "V": 8, "O": 6,  "FF1": 6, "FF2": 4},
}
layers  = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5"]
ops     = ["Q_lin", "K_lin", "V_lin", "Out_lin", "FFN1", "FFN2"]
op_keys = ["Q", "K", "V", "O", "FF1", "FF2"]
prefix  = ["L0", "L1", "L2", "L3", "L4", "L5"]

matrix = np.array([
    [SA_MATRIX_CONFIG[p][o] for o in op_keys]
    for p in prefix
])

fig, ax = plt.subplots(figsize=(11, 5.5))
cmap = plt.cm.get_cmap("RdYlGn", 3)
im   = ax.imshow(matrix, cmap=cmap, vmin=3, vmax=9, aspect="auto")

for i in range(len(layers)):
    for j in range(len(ops)):
        bits  = matrix[i, j]
        color = "white" if bits <= 4 else "black"
        ax.text(j, i, f"{bits}b", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

ax.set_xticks(range(len(ops)))
ax.set_yticks(range(len(layers)))
ax.set_xticklabels(ops, fontsize=11)
ax.set_yticklabels(layers, fontsize=11)
ax.set_title("SA Bit Allocation Across Transformer Layers\n"
             "Dashed border = FAR-locked to 8-bit",
             fontsize=12, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, ticks=[4, 6, 8])
cbar.set_label("Bit Width", fontsize=10)
cbar.ax.set_yticklabels(["4-bit\n(aggressive)", "6-bit\n(moderate)", "8-bit\n(conservative)"])

# Locked layers: L1 FF1/FF2 and L2 FF1/FF2 → col 4,5 row 1,2
for (r, c) in [(1, 4), (1, 5), (2, 4), (2, 5)]:
    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="#1565C0",
                                lw=2.5, linestyle="--"))
# INT8 embedding indicator
ax.set_xlabel("Attention: Q/K/V/Out  |  FFN: Lin1/Lin2     "
              "[Embeddings → INT8 in Hybrid]", fontsize=10)

locked_patch = mpatches.Patch(edgecolor="#1565C0", facecolor="none",
                               linestyle="--", linewidth=2.5,
                               label="FAR-locked (8-bit, sensitivity-based)")
ax.legend(handles=[locked_patch], loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_bit_heatmap.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig2_bit_heatmap.png")

# ===========================================================
# FIGURE 3 — Compression Bar Chart (Theoretical size + ratio)
# ===========================================================
methods_for_bar = [m for m in METHOD_ORDER
                   if get_size(m, "theoretical_mb") is not None]
sizes_theory    = [get_size(m, "theoretical_mb") for m in methods_for_bar]
ratios          = [fp32_size / s for s in sizes_theory]
bar_colors      = [COLORS[m] for m in methods_for_bar]
short_lbls      = [SHORT_LABELS[m] for m in methods_for_bar]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# — Left: absolute size —
bars1 = ax1.bar(short_lbls, sizes_theory, color=bar_colors,
                width=0.6, edgecolor="white", linewidth=1.2)
ax1.axhline(fp32_size, color="black", linestyle="--",
            linewidth=1.4, label=f"FP32 ({fp32_size:.0f} MB)")
ax1.set_ylabel("Theoretical Effective Size (MB)", fontsize=11)
ax1.set_title("Model Size (All Methods)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_ylim(0, fp32_size * 1.2)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, s in zip(bars1, sizes_theory):
    ax1.text(bar.get_x() + bar.get_width() / 2, s + 2,
             f"{s:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# — Right: compression ratio —
bars2 = ax2.bar(short_lbls, ratios, color=bar_colors,
                width=0.6, edgecolor="white", linewidth=1.2)
ax2.axhline(1.0, color="black", linestyle="--",
            linewidth=1.4, label="FP32 baseline (1×)")
ax2.set_ylabel("Compression Ratio (×)", fontsize=11)
ax2.set_title("Compression vs FP32", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(ratios) * 1.25)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, r in zip(bars2, ratios):
    ax2.text(bar.get_x() + bar.get_width() / 2, r + 0.05,
             f"{r:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.suptitle("Compression — All 7 Methods\nFAR + SA Hybrid Quantization Framework",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_compression_bars.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig3_compression_bars.png")

# ===========================================================
# FIGURE 4 — Latency Bar Chart
# ===========================================================
methods_lat = [m for m in METHOD_ORDER if get(m, "latency_avg") is not None]
latencies   = [get(m, "latency_avg") for m in methods_lat]
lat_colors  = [COLORS[m] for m in methods_lat]
lat_labels  = [SHORT_LABELS[m] for m in methods_lat]
lat_p95     = [get(m, "latency_p95", 0) for m in methods_lat]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(lat_labels, latencies, color=lat_colors,
              width=0.55, edgecolor="white", linewidth=1.2)

# P95 error bar
ax.errorbar(range(len(methods_lat)),
            latencies,
            yerr=[np.zeros(len(methods_lat)),
                  [p - a for p, a in zip(lat_p95, latencies)]],
            fmt="none", color="black", capsize=5, linewidth=1.5,
            label="P95 latency")

# 20 ms target line (edge gold standard from paper)
ax.axhline(20.0, color="#F44336", linestyle="--", linewidth=1.5,
           label="20 ms edge target")

ax.set_ylabel("Avg Inference Latency (ms)\nSingle sample, 4-thread CPU", fontsize=11)
ax.set_title("CPU Inference Latency — All Models\n"
             "(Simulated Raspberry Pi 4: CPU-only, 4 threads)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, max(latencies) * 1.3 if latencies else 30)
ax.grid(axis="y", alpha=0.3, linestyle="--")
for bar, lat in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width() / 2, lat + 0.3,
            f"{lat:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_latency_bars.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig4_latency_bars.png")

# ===========================================================
# FIGURE 5 — Accuracy Drop vs FP32 (shows degradation cleanly)
# ===========================================================
methods_acc = [m for m in METHOD_ORDER
               if get(m, "accuracy") is not None and m != "FP32 Baseline"]
acc_drops   = [(fp32_acc - get(m, "accuracy")) * 100 for m in methods_acc]
acc_colors  = [COLORS[m] for m in methods_acc]
acc_labels  = [SHORT_LABELS[m] for m in methods_acc]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(acc_labels, acc_drops, color=acc_colors,
              width=0.55, edgecolor="white", linewidth=1.2)

ax.axhline(1.0, color="#FF9800", linestyle="--", linewidth=1.5,
           label="1% degradation threshold")
ax.axhline(2.0, color="#F44336", linestyle="--", linewidth=1.5,
           label="2% degradation threshold")
ax.axhline(0.0, color="black", linewidth=1.0)

ax.set_ylabel("Accuracy Drop vs FP32 (%)", fontsize=11)
ax.set_title("Accuracy Degradation Relative to FP32 Baseline\n"
             "(Lower is better — negative means improvement)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(min(min(acc_drops) - 0.3, -0.5), max(acc_drops) * 1.4 if acc_drops else 3)
ax.grid(axis="y", alpha=0.3, linestyle="--")
for bar, drop in zip(bars, acc_drops):
    label = f"{drop:+.2f}%"
    va    = "bottom" if drop >= 0 else "top"
    y_off = 0.05 if drop >= 0 else -0.05
    ax.text(bar.get_x() + bar.get_width() / 2, drop + y_off,
            label, ha="center", va=va, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_accuracy_drop.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig5_accuracy_drop.png")

# ----------------------------------------------------------
print(f"\nAll 5 figures saved to {FIGURES_DIR}/")
print("For thesis use: fig1 (main result) + fig2 (SA analysis) + fig3/4/5 (appendix)")
