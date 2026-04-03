from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import baseline_checkpoint_path
from shared.model_workflows import evaluate_pytorch_model, load_classifier_checkpoint, load_sst2_validation_dataset


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="torch")
    model_path = baseline_checkpoint_path()
    model = load_classifier_checkpoint(model_path)
    metrics = evaluate_pytorch_model(model, dataset)
    size_mb = model_path.stat().st_size / (1024**2)

    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1: {metrics['f1']:.4f}")
    print(f"Model File Size: {size_mb:.2f} MB")
    print(f"Average CPU Latency (batch=1): {metrics['latency_ms']:.2f} ms")
    print("\nStage 1 Evaluation Complete.")


if __name__ == "__main__":
    main()
