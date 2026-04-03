import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

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


def main() -> None:
    histories = load_histories()
    if not histories:
        print("No training histories found. Run a training script first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.8))

    for payload in histories:
        model_label = str(payload.get("model_label", "Model"))
        regimen_name = str(payload.get("regimen_name", "regimen"))
        history = payload["history"]

        steps = [entry["global_step"] for entry in history]
        accuracies = [entry["validation_accuracy"] for entry in history]
        best_step = payload.get("best_global_step")
        best_accuracy = payload.get("best_validation_accuracy")

        line_color = COLOR_BY_MODEL.get(model_label, "#37474F")
        line_style = LINESTYLE_BY_REGIMEN.get(regimen_name, "-")
        label = f"{model_label} | {regimen_name}"

        plt.plot(
            steps,
            accuracies,
            label=label,
            color=line_color,
            linestyle=line_style,
            linewidth=2,
            marker="o",
            markersize=4.5,
        )
        if best_step is not None and best_accuracy is not None:
            plt.scatter(best_step, best_accuracy, color=line_color, s=70, zorder=5, edgecolors="white", linewidths=1.0)

    plt.xlabel("Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Training Steps")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "training_convergence_accuracy_vs_steps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
