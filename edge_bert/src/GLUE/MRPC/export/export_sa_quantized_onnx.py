from pathlib import Path
import sys

import torch
from transformers import DistilBertForSequenceClassification


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import ExportWrapper, cleanup_onnx_artifacts, consolidate_onnx_to_single_file
from GLUE.MRPC.quantization import replace_linear_layers
from GLUE.MRPC.task_config import (
    MAX_LENGTH,
    MODEL_NAME,
    NUM_LABELS,
    VOCAB_SIZE,
    baseline_checkpoint_path,
    get_locked_layers,
    load_sa_best_config,
    sa_onnx_path,
)


def main() -> None:
    best_candidate = load_sa_best_config()
    locked_layers = set(get_locked_layers())
    checkpoint_path = baseline_checkpoint_path()

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    model = replace_linear_layers(model, best_candidate, locked_layers=locked_layers)
    wrapped = ExportWrapper(model, mask_dtype=torch.float32).eval()

    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_LENGTH), dtype=torch.long)
    dummy_attention = torch.ones((1, MAX_LENGTH), dtype=torch.long)
    output_path = sa_onnx_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_onnx_artifacts(output_path)

    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_attention),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=False,
        export_params=True,
    )
    consolidate_onnx_to_single_file(output_path)
    print(f"MRPC SA ONNX exported to {output_path}")


if __name__ == "__main__":
    main()

