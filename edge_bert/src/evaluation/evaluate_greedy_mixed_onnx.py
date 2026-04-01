from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="numpy")
    metrics = evaluate_onnx_model(MODELS_DIR / "greedy_quant_model.onnx", dataset)

    print("\n===== GREEDY MIXED PRECISION RESULTS =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Model Size: {metrics['onnx_size_mb']:.2f} MB")
    print(f"Avg CPU Latency: {metrics['latency_avg_ms']:.2f} ms")


if __name__ == "__main__":
    main()
