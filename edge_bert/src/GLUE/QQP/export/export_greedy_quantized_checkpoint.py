from pathlib import Path
import sys

import torch


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.QQP.common import load_classifier_checkpoint
from GLUE.QQP.quantization import build_quantized_checkpoint_state
from GLUE.QQP.task_config import baseline_checkpoint_path, greedy_checkpoint_path, load_greedy_config


def main() -> None:
    bit_config = load_greedy_config()
    model = load_classifier_checkpoint(baseline_checkpoint_path(), device="cpu")
    state_dict = build_quantized_checkpoint_state(model, bit_config)
    output_path = greedy_checkpoint_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"QQP greedy quantized checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
