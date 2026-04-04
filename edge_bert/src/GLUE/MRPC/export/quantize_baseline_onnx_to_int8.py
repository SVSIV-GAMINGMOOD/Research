from pathlib import Path
import sys
import tempfile

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import total_onnx_size_mb
from GLUE.MRPC.task_config import baseline_fp32_onnx_path, baseline_int8_onnx_path


def main() -> None:
    fp32_path = baseline_fp32_onnx_path()
    int8_path = baseline_int8_onnx_path()
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="MRPC.quant.") as tmp_dir:
        sanitized_fp32_path = Path(tmp_dir) / "sanitized_fp32.onnx"
        model = onnx.load(str(fp32_path))
        del model.graph.value_info[:]

        input_batch_param = None
        for inp in model.graph.input:
            tensor_type = inp.type.tensor_type
            if tensor_type.HasField("shape") and len(tensor_type.shape.dim) > 0:
                d0 = tensor_type.shape.dim[0]
                if d0.dim_param:
                    input_batch_param = d0.dim_param
                    break
        if input_batch_param:
            for out in model.graph.output:
                tensor_type = out.type.tensor_type
                if tensor_type.HasField("shape") and len(tensor_type.shape.dim) > 0:
                    d0 = tensor_type.shape.dim[0]
                    if d0.dim_value == 1 and not d0.dim_param:
                        d0.dim_param = input_batch_param
                        d0.dim_value = 0

        onnx.save_model(model, str(sanitized_fp32_path))
        quantize_dynamic(
            model_input=str(sanitized_fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    fp32_size = total_onnx_size_mb(fp32_path)
    int8_size = total_onnx_size_mb(int8_path)
    print(f"MRPC INT8 uniform ONNX saved to {int8_path}")
    print(f"FP32 Size: {fp32_size:.2f} MB")
    print(f"INT8 Size: {int8_size:.2f} MB")
    print(f"Size Reduction: {(1 - int8_size / fp32_size) * 100:.2f}%")


if __name__ == "__main__":
    main()

