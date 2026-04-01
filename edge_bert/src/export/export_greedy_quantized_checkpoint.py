import json
import torch
from pathlib import Path

from transformers import DistilBertForSequenceClassification

SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_best.pt"

with open(MODELS_DIR / "greedy_config.json") as f:
    bit_config = json.load(f)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))

def quantize_tensor(tensor, bits):

    qmin = -(2**(bits-1))
    qmax = (2**(bits-1))-1

    scale = tensor.abs().max()/qmax

    q = torch.round(tensor/scale).clamp(qmin,qmax)

    return q*scale


for name,module in model.named_modules():

    if name in bit_config:

        bits = bit_config[name]

        module.weight.data = quantize_tensor(
            module.weight.data,
            bits
        )

print("Greedy quantization applied")

torch.save(
    model.state_dict(),
    MODELS_DIR / "greedy_quant_model.pt"
)

print(f"Saved: {MODELS_DIR / 'greedy_quant_model.pt'}")
