from pathlib import Path
import sys

import torch
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_BITS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    get_locked_layers,
    load_greedy_config,
    load_sa_best_config,
)


SA_CONFIG = load_sa_best_config()
GREEDY_CONFIG = load_greedy_config()
LOCKED_LAYERS = set(get_locked_layers())
for layer in LOCKED_LAYERS:
    SA_CONFIG[layer] = 8


def compute_size(model, config=None):
    total_bits = 0
    total_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        layer_name = name.replace(".weight", "").replace(".bias", "")
        bits = config.get(layer_name, DEFAULT_BITS) if config else DEFAULT_BITS
        total_bits += params * bits
    return total_bits / 8 / 1024 / 1024, total_params


def main() -> None:
    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )
    total_params = sum(param.numel() for param in model.parameters())
    fp32_size = total_params * 32 / 8 / 1024 / 1024
    int8_size = total_params * 8 / 8 / 1024 / 1024
    greedy_size, _ = compute_size(model, GREEDY_CONFIG)
    sa_size, _ = compute_size(model, SA_CONFIG)

    print("===== EFFECTIVE MODEL SIZE =====\n")
    print(f"FP32 Model        : {fp32_size:.2f} MB")
    print(f"INT8 Model        : {int8_size:.2f} MB")
    print(f"Greedy Mixed      : {greedy_size:.2f} MB")
    print(f"SA Mixed          : {sa_size:.2f} MB")
    print("\nTotal Parameters:", total_params)


if __name__ == "__main__":
    main()
