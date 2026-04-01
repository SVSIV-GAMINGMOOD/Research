from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, export_checkpoint_to_onnx


def main() -> None:
    output_path = export_checkpoint_to_onnx(
        MODELS_DIR / "baseline_best.pt",
        MODELS_DIR / "baseline_fp32.onnx",
    )
    print(f"FP32 ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
