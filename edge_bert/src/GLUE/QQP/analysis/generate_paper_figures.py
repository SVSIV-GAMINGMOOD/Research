from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.QQP.task_config import FIGURES_DIR, RESULTS_DIR, get_locked_layers, load_sa_best_config


TASK_LABEL = "QQP"
MODEL_ORDER = [
    "FP32 Research Reference",
    "Primary Baseline (FP16)",
    "INT8 Uniform",
    "Greedy Mixed",
    "SA (v1)",
    "Hybrid SA (ours)",
]
COLORS = {
    "FP32 Research Reference": "#607D8B",
    "Primary Baseline (FP16)": "#3F51B5",
    "INT8 Uniform": "#2196F3",
    "Greedy Mixed": "#FF9800",
    "SA (v1)": "#9C27B0",
    "Hybrid SA (ours)": "#F44336",
}
SHORT_LABELS = {
    "FP32 Research Reference": "FP32",
    "Primary Baseline (FP16)": "FP16",
    "INT8 Uniform": "INT8",
    "Greedy Mixed": "Greedy",
    "SA (v1)": "SA v1",
    "Hybrid SA (ours)": "Hybrid SA",
}


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_training_histories() -> list[dict]:
    history_root = RESULTS_DIR / "training_histories"
    histories: list[dict] = []
    for regimen_name in ("realistic", "controlled"):
        payload = load_json_if_exists(history_root / regimen_name / "baseline_history.json")
        if payload.get("history"):
            histories.append(payload)
    return histories


def rank_colors(values, *, higher_is_better: bool) -> list:
    values = np.asarray(values, dtype=float)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    if len(values) == 0:
        return []
    if np.allclose(values.max(), values.min()):
        return [cmap(0.5) for _ in values]
    normalized = (values - values.min()) / (values.max() - values.min())
    if not higher_is_better:
        normalized = 1.0 - normalized
    return [cmap(0.12 + 0.76 * score) for score in normalized]


def annotate_bars(ax, bars, fmt: str) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.015 if y_max > y_min else 0.01
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def build_sa_matrix() -> tuple[np.ndarray, list[str], set[str]]:
    sa_config = load_sa_best_config(required=False)
    locked_layers = set(get_locked_layers())
    op_patterns = [
        ("Q", "attention.q_lin"),
        ("K", "attention.k_lin"),
        ("V", "attention.v_lin"),
        ("Out", "attention.out_lin"),
        ("FFN1", "ffn.lin1"),
        ("FFN2", "ffn.lin2"),
    ]
    matrix = []
    for layer_index in range(6):
        row = []
        for _, pattern in op_patterns:
            layer_name = f"distilbert.transformer.layer.{layer_index}.{pattern}"
            row.append(sa_config.get(layer_name, 8))
        matrix.append(row)
    return np.array(matrix), [label for label, _ in op_patterns], locked_layers


def save_tradeoff_figure(all_results: dict) -> bool:
    methods = [name for name in MODEL_ORDER if all_results.get(name)]
    if not methods:
        return False

    sizes = [all_results[name]["onnx_mb"] for name in methods]
    accuracies = [all_results[name]["accuracy"] for name in methods]
    colors = rank_colors(accuracies, higher_is_better=True)

    plt.figure(figsize=(10, 6))
    for index, method in enumerate(methods):
        is_ours = "ours" in method.lower()
        plt.scatter(
            sizes[index],
            accuracies[index],
            s=220 if is_ours else 100,
            c=[colors[index]],
            marker="*" if is_ours else "o",
            edgecolors="white",
            linewidths=1.4,
            zorder=5 if is_ours else 3,
        )
        plt.annotate(method, (sizes[index], accuracies[index]), xytext=(5, 5), textcoords="offset points", fontsize=8.5)
    plt.xlabel("ONNX Model Size (MB)")
    plt.ylabel(f"{TASK_LABEL} Accuracy")
    plt.title(f"{TASK_LABEL} Accuracy vs ONNX Size", fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_tradeoff_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_latency_figure(all_results: dict) -> bool:
    methods = [
        name for name in MODEL_ORDER
        if all_results.get(name) and name != "Primary Baseline (FP16)"
    ]
    if not methods:
        return False

    labels = [SHORT_LABELS[name] for name in methods]
    latencies = [all_results[name]["latency_avg"] for name in methods]
    p95 = [all_results[name].get("latency_p95", value) for name, value in zip(methods, latencies)]
    colors = rank_colors(latencies, higher_is_better=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, latencies, color=colors, width=0.58, edgecolor="white", linewidth=1.1)
    ax.errorbar(
        range(len(methods)),
        latencies,
        yerr=[np.zeros(len(methods)), [max(0.0, hi - lo) for hi, lo in zip(p95, latencies)]],
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.3,
    )
    ax.set_ylabel("Average Inference Latency (ms)")
    ax.set_title(f"{TASK_LABEL} ONNX Latency Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax, bars, "{:.2f}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_latency_bars.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_accuracy_drop_figure(all_results: dict) -> bool:
    if "FP32 Research Reference" not in all_results:
        return False

    fp32_accuracy = all_results["FP32 Research Reference"]["accuracy"]
    methods = [name for name in MODEL_ORDER if all_results.get(name) and name != "FP32 Research Reference"]
    if not methods:
        return False

    labels = [SHORT_LABELS[name] for name in methods]
    drops = [(fp32_accuracy - all_results[name]["accuracy"]) * 100 for name in methods]
    bar_colors = [COLORS.get(name, "#455A64") for name in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, drops, color=bar_colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, color="#FF9800", linestyle="--", linewidth=1.5)
    ax.axhline(2.0, color="#F44336", linestyle="--", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Accuracy Drop vs FP32 (%)")
    ax.set_title("Accuracy Degradation Relative to FP32 Baseline", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax, bars, "{:.2f}%")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_accuracy_drop.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_sa_heatmap() -> bool:
    try:
        matrix, op_labels, locked_layers = build_sa_matrix()
    except FileNotFoundError:
        return False

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(3)
    im = ax.imshow(matrix, cmap=cmap, vmin=3, vmax=9, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            bits = matrix[row_index, col_index]
            layer_name = f"distilbert.transformer.layer.{row_index}." + [
                "attention.q_lin",
                "attention.k_lin",
                "attention.v_lin",
                "attention.out_lin",
                "ffn.lin1",
                "ffn.lin2",
            ][col_index]
            if layer_name in locked_layers:
                ax.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="#1565C0", lw=2.0, linestyle="--"))
            ax.text(col_index, row_index, f"{bits}b", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
    ax.set_xticks(range(len(op_labels)))
    ax.set_yticks(range(6))
    ax.set_xticklabels(op_labels)
    ax.set_yticklabels([f"Layer {index}" for index in range(6)])
    ax.set_title(f"{TASK_LABEL} SA Bit Allocation Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, ticks=[4, 6, 8]).set_label("Bit Width")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_bit_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def save_training_figure() -> bool:
    histories = load_training_histories()
    if not histories:
        return False

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12.6, 5.2), sharex=True)
    fig.subplots_adjust(wspace=0.18)

    regimen_styles = {
        "realistic": {
            "acc_color": "#1565C0",
            "f1_color": "#2E7D32",
            "loss_color": "#EF6C00",
            "linestyle": "-",
            "marker": "o",
            "label": "Realistic",
        },
        "controlled": {
            "acc_color": "#283593",
            "f1_color": "#1B5E20",
            "loss_color": "#BF360C",
            "linestyle": "--",
            "marker": "s",
            "label": "Controlled",
        },
    }

    for payload in histories:
        regimen_name = payload.get("regimen_name", "realistic")
        style = regimen_styles.get(regimen_name, regimen_styles["realistic"])
        history = payload["history"]
        steps = [entry["global_step"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        val_accuracy = [entry["validation_accuracy"] for entry in history]
        val_f1 = [entry.get("validation_f1") for entry in history]
        best_step = payload.get("best_global_step")
        best_accuracy = payload.get("best_validation_accuracy")
        best_epoch = payload.get("best_epoch")
        stopped_early = bool(payload.get("stopped_early"))
        completed_epochs = payload.get("completed_epochs")
        max_epochs = payload.get("max_epochs")
        early_stop_step = (
            steps[-1]
            if stopped_early and completed_epochs is not None and max_epochs is not None and completed_epochs < max_epochs
            else None
        )

        ax_acc.plot(
            steps,
            val_accuracy,
            color=style["acc_color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            label=f"{style['label']} Accuracy",
        )
        if all(value is not None for value in val_f1):
            ax_acc.plot(
                steps,
                val_f1,
                color=style["f1_color"],
                marker="^",
                linestyle=style["linestyle"],
                linewidth=1.9,
                label=f"{style['label']} F1",
            )
        if best_step is not None and best_accuracy is not None:
            ax_acc.scatter(best_step, best_accuracy, color=style["acc_color"], s=65, zorder=6, edgecolors="white", linewidths=1.0)
            ax_acc.annotate(
                f"{style['label']} best: {best_accuracy:.4f} (E{best_epoch})",
                xy=(best_step, best_accuracy),
                xytext=(8, 8 if regimen_name == "realistic" else -18),
                textcoords="offset points",
                fontsize=8.2,
                fontweight="bold",
                color=style["acc_color"],
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": style["acc_color"],
                    "alpha": 0.9,
                },
            )

        ax_loss.plot(
            steps,
            train_loss,
            color=style["loss_color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            label=f"{style['label']} Train Loss",
        )

        if early_stop_step is not None:
            for axis in (ax_acc, ax_loss):
                axis.axvline(early_stop_step, color=style["acc_color"], linestyle=":", linewidth=1.5, alpha=0.85)
            ax_acc.annotate(
                f"{style['label']} early stop",
                xy=(early_stop_step, ax_acc.get_ylim()[1]),
                xytext=(6, -18 if regimen_name == "realistic" else -34),
                textcoords="offset points",
                fontsize=8.2,
                fontweight="bold",
                color=style["acc_color"],
            )
            ax_loss.annotate(
                f"{style['label']} early stop",
                xy=(early_stop_step, ax_loss.get_ylim()[1]),
                xytext=(6, -18 if regimen_name == "realistic" else -34),
                textcoords="offset points",
                fontsize=8.2,
                fontweight="bold",
                color=style["acc_color"],
            )

    for axis in (ax_acc, ax_loss):
        axis.set_xlabel("Cumulative Training Steps")
        axis.ticklabel_format(style="plain", axis="x")
        axis.grid(True, alpha=0.25, linestyle="--")

    ax_acc.set_ylabel("Validation Metric")
    ax_acc.set_title("Validation Convergence", fontweight="bold")
    ax_acc.legend(loc="best")

    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Optimization Trajectory", fontweight="bold")
    ax_loss.legend(loc="best")

    fig.suptitle(f"{TASK_LABEL} Baseline Training Convergence vs Steps", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(FIGURES_DIR / "fig5_training_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_json_if_exists(RESULTS_DIR / "all_model_results.json")

    generated: list[str] = []
    if save_tradeoff_figure(all_results):
        generated.append("fig1_tradeoff_curve.png")
    if save_latency_figure(all_results):
        generated.append("fig2_latency_bars.png")
    if save_accuracy_drop_figure(all_results):
        generated.append("fig3_accuracy_drop.png")
    if save_sa_heatmap():
        generated.append("fig4_bit_heatmap.png")

    if save_training_figure():
        generated.append("fig5_training_convergence.png")

    if not generated:
        raise FileNotFoundError("No figure inputs were available. Run training/evaluation/search steps first.")

    print(f"{TASK_LABEL} figures saved to {FIGURES_DIR}")
    for filename in generated:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
