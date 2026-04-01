import torch
import torch.nn as nn
from pathlib import Path
from transformers import DistilBertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"

# ==========================================================
# BEST CONFIGURATION (Energy 2.1179 | Acc 0.9094)
# ==========================================================

best_candidate = {
    "distilbert.transformer.layer.0.attention.q_lin": 8,
    "distilbert.transformer.layer.0.attention.k_lin": 6,
    "distilbert.transformer.layer.0.attention.v_lin": 4,
    "distilbert.transformer.layer.0.attention.out_lin": 8,
    "distilbert.transformer.layer.0.ffn.lin1": 4,
    "distilbert.transformer.layer.0.ffn.lin2": 6,

    "distilbert.transformer.layer.1.attention.q_lin": 6,
    "distilbert.transformer.layer.1.attention.k_lin": 4,
    "distilbert.transformer.layer.1.attention.v_lin": 4,
    "distilbert.transformer.layer.1.attention.out_lin": 6,

    "distilbert.transformer.layer.2.attention.q_lin": 4,
    "distilbert.transformer.layer.2.attention.k_lin": 6,
    "distilbert.transformer.layer.2.attention.v_lin": 4,
    "distilbert.transformer.layer.2.attention.out_lin": 6,

    "distilbert.transformer.layer.3.attention.q_lin": 8,
    "distilbert.transformer.layer.3.attention.k_lin": 6,
    "distilbert.transformer.layer.3.attention.v_lin": 4,
    "distilbert.transformer.layer.3.attention.out_lin": 8,
    "distilbert.transformer.layer.3.ffn.lin1": 8,
    "distilbert.transformer.layer.3.ffn.lin2": 6,

    "distilbert.transformer.layer.4.attention.q_lin": 8,
    "distilbert.transformer.layer.4.attention.k_lin": 8,
    "distilbert.transformer.layer.4.attention.v_lin": 8,
    "distilbert.transformer.layer.4.attention.out_lin": 4,
    "distilbert.transformer.layer.4.ffn.lin1": 4,
    "distilbert.transformer.layer.4.ffn.lin2": 6,

    "distilbert.transformer.layer.5.attention.q_lin": 4,
    "distilbert.transformer.layer.5.attention.k_lin": 4,
    "distilbert.transformer.layer.5.attention.v_lin": 8,
    "distilbert.transformer.layer.5.attention.out_lin": 6,
    "distilbert.transformer.layer.5.ffn.lin1": 6,
    "distilbert.transformer.layer.5.ffn.lin2": 4,

    "pre_classifier": 4,
    "classifier": 8,
}

# Sensitive layers locked to 8-bit
LOCKED_LAYERS = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1",
}

# ==========================================================
# TRUE WEIGHT QUANTIZATION
# ==========================================================

def quantize_weight(tensor, bits):
    qmin = 0
    qmax = 2**bits - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    zero_point = torch.round(zero_point)

    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor.clamp_(qmin, qmax)

    return q_tensor.to(torch.int32), scale, zero_point


# ==========================================================
# QUANTIZED LINEAR LAYER
# ==========================================================

class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, bits):
        super().__init__()

        weight = linear_layer.weight.data

        q_weight, scale, zero_point = quantize_weight(weight, bits)

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


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.load_state_dict(torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu"))
model.to(DEVICE)
model.eval()

print(f"Loaded {MODELS_DIR / 'baseline_best.pt'}")

# ==========================================================
# REPLACE ALL LINEAR LAYERS
# ==========================================================

for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear):

        if name in LOCKED_LAYERS:
            bits = 8
        else:
            bits = best_candidate.get(name, 8)

        parent = model
        components = name.split(".")
        for comp in components[:-1]:
            parent = getattr(parent, comp)

        setattr(parent, components[-1], QuantizedLinear(module, bits))

        print(f"Replaced {name} with {bits}-bit quantized layer")

print("\nAll Linear layers replaced with TRUE quantized versions.")

# ==========================================================
# EXPORT TO ONNX
# ==========================================================

dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long).to(DEVICE)
dummy_attention = torch.ones((1, 128), dtype=torch.long).to(DEVICE)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention),
    MODELS_DIR / "mixed_precision_real_quant.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=14,
)

print("\nExport complete:")
print(MODELS_DIR / "mixed_precision_real_quant.onnx")
