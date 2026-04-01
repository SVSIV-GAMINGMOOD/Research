from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="numpy")
    metrics = evaluate_onnx_model(MODELS_DIR / "frozen_int8.onnx", dataset)

    print(f"ONNX INT8 Accuracy: {metrics['accuracy']:.4f}")
    print(f"ONNX INT8 F1: {metrics['f1']:.4f}")
    print(f"ONNX INT8 Model Size: {metrics['onnx_size_mb']:.2f} MB")
    print(f"ONNX INT8 Avg CPU Latency: {metrics['latency_avg_ms']:.2f} ms")
    print("\nONNX INT8 Evaluation Complete.")


if __name__ == "__main__":
    main()
