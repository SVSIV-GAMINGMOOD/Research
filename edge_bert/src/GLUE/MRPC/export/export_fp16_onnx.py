from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import compute_fp16_baseline_onnx_metrics, ensure_onnx_provider, export_checkpoint_to_onnx
from GLUE.MRPC.task_config import FP16_ONNX_PROVIDER, baseline_checkpoint_path, baseline_fp16_onnx_path


def main() -> None:
    ensure_onnx_provider(FP16_ONNX_PROVIDER)
    output_path = export_checkpoint_to_onnx(
        baseline_checkpoint_path(),
        baseline_fp16_onnx_path(),
        use_half=True,
    )
    print(f"MRPC FP16 baseline ONNX exported to {output_path}")

    metrics = compute_fp16_baseline_onnx_metrics(overwrite=True)
    print(f"FP16 baseline accuracy: {metrics['accuracy']:.4f}")
    print(f"FP16 baseline latency: {metrics['latency_avg_ms']:.2f} ms")
    print(f"FP16 baseline ONNX size: {metrics['onnx_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()

