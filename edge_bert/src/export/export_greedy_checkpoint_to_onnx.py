import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification

SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"
MODEL_PATH = MODELS_DIR / "greedy_quant_model.pt"
ONNX_PATH = MODELS_DIR / "greedy_quant_model.onnx"

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=14
)

print("Greedy ONNX exported:", ONNX_PATH)
