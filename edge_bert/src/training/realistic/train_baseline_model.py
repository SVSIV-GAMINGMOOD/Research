from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.training_workflows import TrainingRunConfig, run_training_experiment


def main() -> None:
    run_training_experiment(
        TrainingRunConfig(
            regimen_name="realistic",
            model_label="Baseline",
            checkpoint_name="baseline_best.pt",
            history_file_name="baseline_history.json",
            max_epochs=20,
            early_stopping_patience=2,
        )
    )


if __name__ == "__main__":
    main()
