from pathlib import Path
import sys

import torch
from transformers import DistilBertForSequenceClassification


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import ExportWrapper, cleanup_onnx_artifacts, consolidate_onnx_to_single_file, total_onnx_size_mb
from GLUE.MRPC.quantization import QuantizedEmbedding, QuantizedLinear, replace_embedding_layers, replace_linear_layers
from GLUE.MRPC.task_config import (
    MAX_LENGTH,
    MODEL_NAME,
    NUM_LABELS,
    QUANTIZATION_SETTINGS,
    VOCAB_SIZE,
    baseline_checkpoint_path,
    get_locked_layers,
    hybrid_onnx_path,
    hybrid_summary_path,
    load_sa_best_config,
    save_json,
)


def main() -> None:
    sa_config = load_sa_best_config()
    locked_layers = set(get_locked_layers())
    checkpoint_path = baseline_checkpoint_path()

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    model = replace_embedding_layers(model)
    model = replace_linear_layers(model, sa_config, locked_layers=locked_layers)

    total_bits = 0
    total_params = 0
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            params = module.q_weight.numel()
            total_bits += params * module.bits
            total_params += params
        elif isinstance(module, QuantizedEmbedding):
            params = module.q_weight.numel()
            total_bits += params * QUANTIZATION_SETTINGS["embedding_bits"]
            total_params += params

    fp32_params = sum(p.numel() for p in model.parameters())
    total_bits += fp32_params * 32
    total_params += fp32_params
    effective_bits = total_bits / total_params
    effective_size = total_bits / 8 / 1024 / 1024
    fp32_size = total_params * 32 / 8 / 1024 / 1024

    wrapped = ExportWrapper(model, mask_dtype=torch.float32).eval()
    dummy_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long)
    output_path = hybrid_onnx_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_onnx_artifacts(output_path)

    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=False,
        export_params=True,
    )
    consolidate_onnx_to_single_file(output_path)

    summary = {
        "strategy": {
            "embeddings": f"INT{QUANTIZATION_SETTINGS['embedding_bits']} (per-row scale)",
            "locked_layers": "8-bit",
            "sa_layers": "mixed {4,6,8} bit",
            "layernorm": "FP32",
            "softmax": "FP32",
        },
        "sizes": {
            "fp32_mb": round(fp32_size, 2),
            "effective_mb": round(effective_size, 2),
            "onnx_mb": round(total_onnx_size_mb(output_path), 2),
            "compression_ratio": round(fp32_size / effective_size, 2),
            "effective_bits": round(effective_bits, 2),
        },
        "locked_layers": sorted(locked_layers),
        "sa_config": sa_config,
    }
    save_json(hybrid_summary_path(), summary)
    print(f"MRPC hybrid ONNX exported to {output_path}")
    print(f"MRPC hybrid summary saved to {hybrid_summary_path()}")


if __name__ == "__main__":
    main()

