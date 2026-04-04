import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import FIGURES_DIR, RESULTS_DIR

COLOR_BY_MODEL = {
    "Baseline": "#1565C0",
    "Frozen": "#2E7D32",
}

LINESTYLE_BY_REGIMEN = {
    "realistic": "-",
    "controlled": "--",
}

MARKER_BY_REGIMEN = {
    "realistic": "o",
    "controlled": "s",
}

REGIMEN_LABEL = {
    "realistic": "Realistic (early stopping)",
    "controlled": "Controlled (fixed 10 epochs)",
}


def load_histories() -> list[dict[str, object]]:
    history_root = RESULTS_DIR / "training_histories"
    histories: list[dict[str, object]] = []
    if not history_root.exists():
        return histories

    for path in sorted(history_root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("history"):
            histories.append(payload)
    return histories


def style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_best_annotation(ax, step: int, value: float, color: str, label: str, y_pad: float) -> None:
    ax.annotate(
        label,
        xy=(step, value),
        xytext=(8, y_pad),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.9,
        },
    )


def main() -> None:
    histories = load_histories()
    if not histories:
        print("No training histories found. Run a training script first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "font.family": "DejaVu Sans",
        }
    )

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    fig.subplots_adjust(wspace=0.16)

    max_step = 0
    acc_values: list[float] = []
    loss_values: list[float] = []

    ordered_histories = sorted(
        histories,
        key=lambda payload: (
            str(payload.get("regimen_name", "")) != "realistic",
            str(payload.get("model_label", "")),
        ),
    )

    for payload in ordered_histories:
        model_label = str(payload.get("model_label", "Model"))
        regimen_name = str(payload.get("regimen_name", "regimen"))
        history = payload["history"]

        steps = [entry["global_step"] for entry in history]
        accuracies = [entry["validation_accuracy"] for entry in history]
        losses = [entry["train_loss"] for entry in history]
        best_step = payload.get("best_global_step")
        best_accuracy = payload.get("best_validation_accuracy")
        best_epoch = payload.get("best_epoch")
        completed_epochs = payload.get("completed_epochs")
        max_epochs = payload.get("max_epochs")

        line_color = COLOR_BY_MODEL.get(model_label, "#37474F")
        line_style = LINESTYLE_BY_REGIMEN.get(regimen_name, "-")
        marker = MARKER_BY_REGIMEN.get(regimen_name, "o")
        label = f"{model_label} | {REGIMEN_LABEL.get(regimen_name, regimen_name)}"

        ax_acc.plot(
            steps,
            accuracies,
            label=label,
            color=line_color,
            linestyle=line_style,
            linewidth=2.4,
            marker=marker,
            markersize=5.3,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        ax_loss.plot(
            steps,
            losses,
            color=line_color,
            linestyle=line_style,
            linewidth=2.2,
            marker=marker,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
        if best_step is not None and best_accuracy is not None:
            ax_acc.scatter(best_step, best_accuracy, color=line_color, s=86, zorder=6, edgecolors="white", linewidths=1.2)
            add_best_annotation(
                ax_acc,
                int(best_step),
                float(best_accuracy),
                line_color,
                f"Best: {best_accuracy:.4f} (E{best_epoch})",
                10 if model_label == "Baseline" else -18,
            )

        if regimen_name == "realistic" and completed_epochs is not None and max_epochs is not None and completed_epochs < max_epochs:
            final_step = steps[-1]
            ax_acc.axvline(final_step, color=line_color, linestyle=":", linewidth=1.1, alpha=0.55)
            ax_loss.axvline(final_step, color=line_color, linestyle=":", linewidth=1.1, alpha=0.55)

        max_step = max(max_step, max(steps))
        acc_values.extend(accuracies)
        loss_values.extend(losses)

    style_axes(ax_acc, "Validation Accuracy")
    style_axes(ax_loss, "Training Loss")

    acc_min = min(acc_values)
    acc_max = max(acc_values)
    loss_min = min(loss_values)
    loss_max = max(loss_values)

    ax_acc.set_ylim(acc_min - 0.004, acc_max + 0.006)
    ax_loss.set_ylim(max(0.0, loss_min - 0.01), loss_max + 0.025)
    ax_acc.set_xlim(0, max_step * 1.03)
    ax_loss.set_xlim(0, max_step * 1.03)

    for axis in (ax_acc, ax_loss):
        axis.set_xlabel("Cumulative Training Steps", fontweight="bold")
        axis.ticklabel_format(style="plain", axis="x")

    ax_acc.set_title("Validation Convergence", fontweight="bold", pad=12)
    ax_loss.set_title("Optimization Trajectory", fontweight="bold", pad=12)
    acc_legend = ax_acc.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#CFD8DC",
        borderpad=0.4,
        handlelength=2.0,
        labelspacing=0.35,
    )
    loss_handles = [
        plt.Line2D(
            [0],
            [0],
            color=COLOR_BY_MODEL["Baseline"],
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=2.2,
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            color=COLOR_BY_MODEL["Frozen"],
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=2.2,
            label="Frozen",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#546E7A",
            linestyle="-",
            linewidth=2.0,
            label="Realistic",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#546E7A",
            linestyle="--",
            linewidth=2.0,
            label="Controlled",
        ),
    ]
    loss_legend = ax_loss.legend(
        handles=loss_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#CFD8DC",
        borderpad=0.4,
        handlelength=2.0,
        labelspacing=0.35,
    )
    acc_legend.set_zorder(10)
    loss_legend.set_zorder(10)

    fig.suptitle(
        "Training Convergence Under Realistic and Controlled Regimens",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.92, wspace=0.18)

    output_path = FIGURES_DIR / "training_convergence_accuracy_vs_steps.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
