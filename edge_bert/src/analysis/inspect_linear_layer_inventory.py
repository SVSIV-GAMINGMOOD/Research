import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification
import sys
from shared.model_workflows import load_classifier_checkpoint
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS, baseline_checkpoint_path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# =========================
# Load Full Fine-Tuned Model
# =========================
model = load_classifier_checkpoint(
    baseline_checkpoint_path(),
    device="cpu"
)

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
