from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.training_workflows import TrainingRunConfig, run_training_experiment


def main() -> None:
    run_training_experiment(
        TrainingRunConfig(
            regimen_name="realistic",
            model_label="Frozen",
            checkpoint_name="frozen_best.pt",
            history_file_name="frozen_history.json",
            max_epochs=20,
            early_stopping_patience=2,
            freeze_embeddings=True,
            freeze_transformer_layers=2,
        )
    )


if __name__ == "__main__":
    main()
