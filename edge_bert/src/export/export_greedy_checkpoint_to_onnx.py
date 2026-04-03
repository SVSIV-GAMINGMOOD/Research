import torch
from pathlib import Path
import sys
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS

SRC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = SRC_ROOT / "models"
MODEL_PATH = MODELS_DIR / "greedy_quant_model.pt"
ONNX_PATH = MODELS_DIR / "greedy_quant_model.onnx"

class GreedyExportWrapper(torch.nn.Module):
    """
    Export-friendly wrapper.

    Transformers' DistilBERT builds an attention mask internally using `sdpa_mask`.
    During TorchScript-based ONNX tracing, that code path can crash with a shape edge case
    (IndexError inside `q_length[0]`).

    To avoid that, we compute a 4D attention mask from the 2D `attention_mask` input and pass
    it into the underlying model. When a 4D mask is provided, Transformers bypasses its
    bidirectional-mask creation.
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


model = DistilBertForSequenceClassification.from_pretrained(
    DEFAULT_MODEL_NAME,
    num_labels=DEFAULT_NUM_LABELS,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

wrapped = GreedyExportWrapper(model)
wrapped.eval()

dummy_input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

torch.onnx.export(
    wrapped,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=18,
    dynamo=False,
)

print("Greedy ONNX exported:", ONNX_PATH)
