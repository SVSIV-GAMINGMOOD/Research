from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.task_config import MODELS_DIR, RESULTS_DIR, training_config
from GLUE.MRPC.training_workflows import TrainingRunConfig, run_training_experiment


def main() -> None:
    regimen = "realistic"
    config = training_config(regimen)
    run_training_experiment(
        TrainingRunConfig(
            regimen_name=regimen,
            model_label="MRPC Baseline",
            checkpoint_name="baseline_best.pt",
            history_file_name="baseline_history.json",
            max_epochs=config["max_epochs"],
            early_stopping_patience=config["early_stopping_patience"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            seed=config["seed"],
            models_dir=MODELS_DIR,
            results_dir=RESULTS_DIR,
        )
    )


if __name__ == "__main__":
    main()

