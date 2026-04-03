import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_EMBEDDING_BITS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    MODELS_DIR,
    baseline_checkpoint_path,
    get_locked_layers,
    load_sa_best_config,
)


DEVICE = "cpu"
BASELINE_PATH = baseline_checkpoint_path()
ONNX_OUT = MODELS_DIR / "hybrid_quant_model.onnx"
SUMMARY_OUT = MODELS_DIR / "hybrid_config_summary.json"
SEQ_LEN = 128
SA_CONFIG = load_sa_best_config()
LOCKED_LAYERS = set(get_locked_layers())
for layer in LOCKED_LAYERS:
    SA_CONFIG[layer] = 8


class ExportWrapper(torch.nn.Module):
    """
    Export-friendly wrapper.

    Transformers 5.5 builds a bidirectional attention mask internally via `sdpa_mask`, which can
    crash under TorchScript-based ONNX tracing (IndexError inside `q_length[0]`).
    If we provide a precomputed 4D attention mask, Transformers bypasses that path.
    """

    def __init__(self, model: DistilBertForSequenceClassification):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: (batch, seq) with 1 for real tokens and 0 for padding
        attn_bool = attention_mask > 0
        # Pairwise token-visibility mask: (batch, 1, seq, seq)
        pairwise = attn_bool[:, None, :, None] & attn_bool[:, None, None, :]

        min_val = torch.finfo(torch.float32).min
        mask = torch.where(pairwise, torch.zeros_like(pairwise, dtype=torch.float32), min_val)

        out = self.model(input_ids=input_ids, attention_mask=mask)
        return out.logits


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear_layer.weight.data
        q_min = 0
        q_max = 2**bits - 1
        w_min = weight.min()
        w_max = weight.max()
        scale = (w_max - w_min) / (q_max - q_min + 1e-8)
        zero_point = torch.round(q_min - w_min / (scale + 1e-8))
        q_weight = torch.clamp(torch.round(weight / scale + zero_point), q_min, q_max).to(torch.int32)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return nn.functional.linear(x, weight, self.bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding):
        super().__init__()
        weight = embedding_layer.weight.data
        q_min, q_max = -128, 127
        abs_max = weight.abs().max(dim=1, keepdim=True).values + 1e-8
        scale = abs_max / 127.0
        q_weight = torch.clamp(torch.round(weight / scale), q_min, q_max).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.num_embeddings = embedding_layer.num_embeddings
        self.embedding_dim = embedding_layer.embedding_dim
        self.padding_idx = embedding_layer.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        q_rows = self.q_weight[input_ids]
        scales = self.scale[input_ids]
        return q_rows.float() * scales


def replace_linear_layers(model: nn.Module, config: dict) -> nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            bits = config.get(name, 8)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedLinear(module, bits))
            print(f"  [LINEAR ] {name:<60} -> {bits}-bit")
    return model


def replace_embedding_layers(model: nn.Module) -> nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Embedding):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedEmbedding(module))
            print(f"  [EMBED  ] {name:<60} -> INT{DEFAULT_EMBEDDING_BITS}")
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )
    model.load_state_dict(torch.load(BASELINE_PATH, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()

    model = replace_embedding_layers(model)
    model = replace_linear_layers(model, SA_CONFIG)

    total_bits = 0
    total_params = 0
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            params = module.q_weight.numel()
            total_bits += params * module.bits
            total_params += params
        elif isinstance(module, QuantizedEmbedding):
            params = module.q_weight.numel()
            total_bits += params * DEFAULT_EMBEDDING_BITS
            total_params += params

    fp32_params = sum(p.numel() for name, p in model.named_parameters() if "q_weight" not in name)
    total_bits += fp32_params * 32
    total_params += fp32_params
    effective_bits = total_bits / total_params
    effective_size = total_bits / 8 / 1024 / 1024
    fp32_size = total_params * 32 / 8 / 1024 / 1024

    dummy_ids = torch.randint(0, 30522, (1, SEQ_LEN), dtype=torch.long, device=DEVICE)
    dummy_mask = torch.ones((1, SEQ_LEN), dtype=torch.long, device=DEVICE)

    # Wrap the model to precompute the 4D attention mask, bypassing the
    # sdpa_mask crash in Transformers 5.5 during TorchScript-based ONNX tracing.
    wrapped = ExportWrapper(model).to(DEVICE).eval()

    torch.onnx.export(
        wrapped,                          # <-- wrapped model, not raw model
        (dummy_ids, dummy_mask),
        ONNX_OUT,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx_size = os.path.getsize(ONNX_OUT) / 1024 / 1024

    summary = {
        "strategy": {
            "embeddings": f"INT{DEFAULT_EMBEDDING_BITS} (per-row scale)",
            "locked_layers": "8-bit (sensitivity threshold)",
            "sa_layers": "mixed {4,6,8} bit",
            "layernorm": "FP32",
            "softmax": "FP32",
        },
        "sizes": {
            "fp32_mb": round(fp32_size, 2),
            "effective_mb": round(effective_size, 2),
            "onnx_mb": round(onnx_size, 2),
            "compression_ratio": round(fp32_size / effective_size, 2),
            "effective_bits": round(effective_bits, 2),
        },
        "locked_layers": list(LOCKED_LAYERS),
        "sa_config": SA_CONFIG,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nConfig summary saved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()