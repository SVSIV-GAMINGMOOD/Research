from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import MODELS_DIR, baseline_checkpoint_path
from shared.model_workflows import export_checkpoint_to_onnx


def main() -> None:
    output_path = export_checkpoint_to_onnx(
        baseline_checkpoint_path(),
        MODELS_DIR / "baseline_fp32.onnx",
    )
    print(f"FP32 ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
