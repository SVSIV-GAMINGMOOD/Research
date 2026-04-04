from __future__ import annotations

import copy
from pathlib import Path
import sys

import torch
from sklearn.metrics import accuracy_score


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import ensure_cuda_available, load_classifier_checkpoint, load_mrpc_dataset
from GLUE.MRPC.quantization import fake_quantize_tensor
from GLUE.MRPC.task_config import SENSITIVITY_SETTINGS, baseline_checkpoint_path, derive_locked_layers, save_json, sensitivity_results_path


BIT_SIMULATION = SENSITIVITY_SETTINGS["bit_simulation"]
LOCK_THRESHOLD = SENSITIVITY_SETTINGS["lock_threshold"]


def evaluate(model, dataset, *, device: str) -> float:
    predictions = []
    labels = []
    with torch.no_grad():
        for sample in dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.append(torch.argmax(logits, dim=-1).item())
            labels.append(sample["labels"].item())
    return accuracy_score(labels, predictions)


def main() -> None:
    ensure_cuda_available("MRPC sensitivity analysis")
    device = "cuda"
    subset_size = SENSITIVITY_SETTINGS["subset_size"]
    subset_seed = SENSITIVITY_SETTINGS["subset_seed"]

    val_dataset = load_mrpc_dataset(
        format_type="torch",
        max_samples=subset_size,
        shuffle=True,
        shuffle_seed=subset_seed,
    )
    baseline_model = load_classifier_checkpoint(
        baseline_checkpoint_path(),
        device=device,
        use_half=True,
    )
    baseline_accuracy = evaluate(baseline_model, val_dataset, device=device)
    print(f"MRPC sensitivity baseline accuracy (FP16 subset): {baseline_accuracy:.4f}")

    fp32_model = load_classifier_checkpoint(
        baseline_checkpoint_path(),
        device=device,
        use_half=False,
    )
    sensitivities = []
    for name, module in fp32_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            model_copy = copy.deepcopy(fp32_model)
            target_layer = dict(model_copy.named_modules())[name]
            target_layer.weight.data = fake_quantize_tensor(target_layer.weight.data, BIT_SIMULATION)
            accuracy = evaluate(model_copy, val_dataset, device=device)
            drop = baseline_accuracy - accuracy
            sensitivities.append((name, drop))
            print(f"{name} -> Accuracy Drop: {drop:.4f}")

    sensitivities.sort(key=lambda item: item[1], reverse=True)
    sensitivity_by_layer = {layer: round(drop, 4) for layer, drop in sensitivities}
    locked_layers = derive_locked_layers(sensitivity_by_layer, threshold=LOCK_THRESHOLD)

    payload = {
        "model_name": "distilbert-base-uncased",
        "dataset_name": "glue",
        "dataset_config": "mrpc",
        "bit_simulation": BIT_SIMULATION,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "lock_threshold": LOCK_THRESHOLD,
        "subset_policy": {
            "strategy": SENSITIVITY_SETTINGS["subset_strategy"],
            "size": subset_size,
            "seed": subset_seed,
        },
        "locked_layers": locked_layers,
        "sensitivity_by_layer": sensitivity_by_layer,
        "sorted_layers": [
            {"layer": layer, "accuracy_drop": round(drop, 4)}
            for layer, drop in sensitivities
        ],
    }
    save_json(sensitivity_results_path(), payload)

    print("\nMost sensitive MRPC layers:")
    for layer, drop in sensitivities[:10]:
        print(layer, round(drop, 4))


if __name__ == "__main__":
    main()

