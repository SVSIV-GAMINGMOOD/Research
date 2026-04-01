import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS, MODELS_DIR, get_locked_layers, load_sa_best_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_CANDIDATE = load_sa_best_config()
LOCKED_LAYERS = set(get_locked_layers())


def quantize_weight(tensor: torch.Tensor, bits: int):
    qmin = 0
    qmax = 2**bits - 1
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
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
    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )
    model.load_state_dict(torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {MODELS_DIR / 'baseline_best.pt'}")

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            bits = 8 if name in LOCKED_LAYERS else BEST_CANDIDATE.get(name, 8)
            parent = model
            components = name.split(".")
            for comp in components[:-1]:
                parent = getattr(parent, comp)
            setattr(parent, components[-1], QuantizedLinear(module, bits))
            print(f"Replaced {name} with {bits}-bit quantized layer")

    dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long).to(DEVICE)
    dummy_attention = torch.ones((1, 128), dtype=torch.long).to(DEVICE)
    output_path = MODELS_DIR / "mixed_precision_real_quant.onnx"
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=14,
    )
    print("\nExport complete:")
    print(output_path)


if __name__ == "__main__":
    main()
