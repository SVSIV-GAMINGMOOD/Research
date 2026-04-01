import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "baseline_best.pt"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load your custom trained weights
state = torch.load(BASELINE_PATH, map_location="cpu")
model.load_state_dict(state)

# THE FIX: Isolate the base model
base_model = model.distilbert

# Forcefully remove the problematic "cls" (classification/vocab) layers from the state dictionary
state_dict = base_model.state_dict()
keys_to_delete = []

for key in state_dict.keys():
    # If a layer has "vocab" or "classifier" or "pre_classifier" in the name, mark it for deletion
    if "vocab" in key or "classifier" in key or "pre_classifier" in key:
        keys_to_delete.append(key)

for key in keys_to_delete:
    print(f"Deleting problematic layer to appease llama.cpp: {key}")
    del state_dict[key]

# Load the cleaned state dict back into the base model
# strict=False is required because we just deleted layers
base_model.load_state_dict(state_dict, strict=False)

# Save the perfectly clean model in the standard Hugging Face format
out_dir = MODELS_DIR / "distilbert_hf_format"
base_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print(f"\nSuccess! Cleaned base model saved to {out_dir}")
