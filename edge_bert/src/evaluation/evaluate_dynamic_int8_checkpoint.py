import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, evaluate_pytorch_model, load_classifier_checkpoint, load_sst2_validation_dataset


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="torch")
    model = load_classifier_checkpoint(MODELS_DIR / "baseline_best.pt")
    model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    output_path = MODELS_DIR / "baseline_int8.pt"
    torch.save(model_int8.state_dict(), output_path)
    metrics = evaluate_pytorch_model(model_int8, dataset)
    size_mb = output_path.stat().st_size / (1024**2)

    print("Dynamic INT8 quantization applied.")
    print(f"INT8 Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"INT8 Validation F1: {metrics['f1']:.4f}")
    print(f"INT8 Model File Size: {size_mb:.2f} MB")
    print(f"INT8 Average CPU Latency (batch=1): {metrics['latency_ms']:.2f} ms")
    print("\nStage 2 Complete.")


if __name__ == "__main__":
    main()
