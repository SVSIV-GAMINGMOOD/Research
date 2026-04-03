import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    MODELS_DIR,
    baseline_checkpoint_path,
    get_locked_layers,
    load_sa_best_config,
)


# Export on CPU to avoid device-mismatch issues during ONNX tracing.
EXPORT_DEVICE = torch.device("cpu")
BEST_CANDIDATE = load_sa_best_config()
LOCKED_LAYERS = set(get_locked_layers())


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


def quantize_weight(tensor: torch.Tensor, bits: int):
    qmin = 0
    qmax = 2**bits - 1
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    scale = scale + 1e-8
    zero_point = torch.round(qmin - min_val / (scale + 1e-8))
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor.clamp_(qmin, qmax)
    return q_tensor.to(torch.int32), scale, zero_point


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        q_weight, scale, zero_point = quantize_weight(linear_layer.weight.data, bits)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return torch.nn.functional.linear(x, weight, self.bias)


def main() -> None:
    checkpoint_path = baseline_checkpoint_path()
    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"Loaded {checkpoint_path}")

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            bits = 8 if name in LOCKED_LAYERS else BEST_CANDIDATE.get(name, 8)
            parent = model
            components = name.split(".")
            for comp in components[:-1]:
                parent = getattr(parent, comp)
            setattr(parent, components[-1], QuantizedLinear(module, bits))
            print(f"Replaced {name} with {bits}-bit quantized layer")

    # Move the fully-patched model (including quantized buffers) onto the export device.
    model.to(EXPORT_DEVICE)
    wrapped = ExportWrapper(model).to(EXPORT_DEVICE).eval()

    dummy_input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long, device=EXPORT_DEVICE)
    dummy_attention = torch.ones((1, 128), dtype=torch.long, device=EXPORT_DEVICE)
    output_path = MODELS_DIR / "mixed_precision_real_quant.onnx"
    torch.onnx.export(
    wrapped,
    (dummy_input_ids, dummy_attention),
    output_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=18,
    dynamo=False,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    },
)
    print("\nExport complete:")
    print(output_path)


if __name__ == "__main__":
    main()
