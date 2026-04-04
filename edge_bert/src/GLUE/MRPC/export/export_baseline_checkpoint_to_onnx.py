from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import export_checkpoint_to_onnx
from GLUE.MRPC.task_config import baseline_checkpoint_path, baseline_fp32_onnx_path


def main() -> None:
    output_path = export_checkpoint_to_onnx(
        baseline_checkpoint_path(),
        baseline_fp32_onnx_path(),
        use_half=False,
    )
    print(f"MRPC FP32 reference ONNX exported to {output_path}")


if __name__ == "__main__":
    main()

