import json
import torch
from pathlib import Path
import sys

from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS, MODELS_DIR, baseline_checkpoint_path

SRC_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = baseline_checkpoint_path()

with open(MODELS_DIR / "greedy_config.json") as f:
    bit_config = json.load(f)

model = DistilBertForSequenceClassification.from_pretrained(
    DEFAULT_MODEL_NAME,
    num_labels=DEFAULT_NUM_LABELS
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
