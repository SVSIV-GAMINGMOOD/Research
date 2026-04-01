from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, quantize_onnx_model


def main() -> None:
    stats = quantize_onnx_model(
        MODELS_DIR / "baseline_fp32.onnx",
        MODELS_DIR / "baseline_int8.onnx",
    )
    print("ONNX Dynamic INT8 model saved to models/baseline_int8.onnx")
    print(f"FP32 Size: {stats['fp32_size_mb']:.2f} MB")
    print(f"INT8 Size: {stats['int8_size_mb']:.2f} MB")
    print(f"Size Reduction: {stats['size_reduction_pct']:.2f}%")


if __name__ == "__main__":
    main()
