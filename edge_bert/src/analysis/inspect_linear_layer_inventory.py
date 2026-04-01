import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# =========================
# Load Full Fine-Tuned Model
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    DEFAULT_MODEL_NAME,
    num_labels=DEFAULT_NUM_LABELS
)
model.load_state_dict(torch.load(MODELS_DIR / "baseline_best.pt", map_location="cpu"))
model.eval()

# =========================
# Extract Linear Layers
# =========================
linear_layers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append(name)

print("\nQuantizable Linear Layers:\n")

for idx, layer in enumerate(linear_layers):
    print(f"{idx}: {layer}")

print(f"\nTotal Linear Layers: {len(linear_layers)}")
