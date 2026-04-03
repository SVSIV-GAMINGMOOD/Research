import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS, baseline_checkpoint_path

MODEL_NAME = DEFAULT_MODEL_NAME
SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"
BASELINE_PATH = baseline_checkpoint_path()

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=DEFAULT_NUM_LABELS)

# Load your custom trained weights
state = torch.load(BASELINE_PATH, map_location="cpu")
model.load_state_dict(state)

# THE FIX: Isolate the base model
base_model = model.distilbert

# Forcefully remove the problematic "cls" (classification/vocab) layers from the state dictionary
state_dict = {
    k: v for k, v in model.state_dict().items()
    if not any(x in k for x in ["vocab", "classifier", "pre_classifier"])
}

# Load the cleaned state dict back into the base model
# strict=False is required because we just deleted layers
base_model.load_state_dict(state_dict, strict=False)

# Save the perfectly clean model in the standard Hugging Face format
out_dir = MODELS_DIR / "distilbert_hf_format"
base_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print(f"\nSuccess! Cleaned base model saved to {out_dir}")
