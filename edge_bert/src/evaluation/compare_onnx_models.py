from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


MODELS = {
    "FP32": MODELS_DIR / "baseline_fp32.onnx",
    "INT8": MODELS_DIR / "baseline_int8.onnx",
    "Frozen_INT8": MODELS_DIR / "frozen_int8.onnx",
    "Greedy_Mixed": MODELS_DIR / "greedy_quant_model.onnx",
    "SA_Mixed": MODELS_DIR / "mixed_precision_real_quant.onnx",
}


def main() -> None:
    dataset = load_sst2_validation_dataset(format_type="numpy")
    results = []
    print("\n===== MODEL COMPARISON =====\n")

    for name, model_path in MODELS.items():
        print(f"Evaluating {name} ...")
        metrics = evaluate_onnx_model(model_path, dataset)
        results.append((name, metrics))
        print(
            f"{name} -> Acc {metrics['accuracy']:.4f} | F1 {metrics['f1']:.4f} "
            f"| Size {metrics['onnx_size_mb']:.2f}MB | Lat {metrics['latency_avg_ms']:.2f}ms"
        )

    print("\n===== FINAL RESULTS TABLE =====\n")
    for name, metrics in results:
        print(
            f"{name:12} | Acc {metrics['accuracy']:.4f} | F1 {metrics['f1']:.4f} "
            f"| Size {metrics['onnx_size_mb']:.2f}MB | Lat {metrics['latency_avg_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
