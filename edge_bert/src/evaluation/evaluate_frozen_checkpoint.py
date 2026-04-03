from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import frozen_checkpoint_path
from shared.model_workflows import evaluate_pytorch_model, load_classifier_checkpoint, load_sst2_validation_dataset


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="torch")
    model = load_classifier_checkpoint(frozen_checkpoint_path())
    metrics = evaluate_pytorch_model(model, dataset)

    print(f"Frozen FP32 Accuracy: {metrics['accuracy']:.4f}")
    print(f"Frozen FP32 F1: {metrics['f1']:.4f}")
    print(f"Frozen FP32 Latency: {metrics['latency_ms']:.2f} ms")
    print("\nFrozen FP32 Evaluation Complete.")


if __name__ == "__main__":
    main()
