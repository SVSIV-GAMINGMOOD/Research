import copy
from pathlib import Path
import sys
import torch
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    SENSITIVITY_SETTINGS,
    baseline_checkpoint_path,
    derive_locked_layers,
    save_sensitivity_results,
)
from shared.model_workflows import MODELS_DIR, load_classifier_checkpoint, load_sst2_validation_dataset


BIT_SIMULATION = SENSITIVITY_SETTINGS["bit_simulation"]
LOCK_THRESHOLD = SENSITIVITY_SETTINGS["lock_threshold"]


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = 0.0
    qmax = 2.0**bits - 1.0
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    q_tensor = zero_point + tensor / (scale + 1e-8)
    q_tensor.clamp_(qmin, qmax).round_()
    return scale * (q_tensor - zero_point)


def evaluate(model, dataset) -> float:
    predictions = []
    true_labels = []
    with torch.no_grad():
        for sample in dataset:
            input_ids = sample["input_ids"].unsqueeze(0)
            attention_mask = sample["attention_mask"].unsqueeze(0)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(torch.argmax(outputs.logits, dim=-1).item())
            true_labels.append(sample["labels"].item())
    return accuracy_score(true_labels, predictions)


def main() -> None:
    val_dataset = load_sst2_validation_dataset(format_type="torch")
    base_model = load_classifier_checkpoint(
    baseline_checkpoint_path(),
    device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    baseline_acc = evaluate(base_model, val_dataset)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")

    sensitivities = []
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            model_copy = copy.deepcopy(base_model)
            target_layer = dict(model_copy.named_modules())[name]
            target_layer.weight.data = fake_quantize_tensor(target_layer.weight.data, BIT_SIMULATION)
            acc = evaluate(model_copy, val_dataset)
            drop = baseline_acc - acc
            sensitivities.append((name, drop))
            print(f"{name} -> Accuracy Drop: {drop:.4f}")

    sensitivities.sort(key=lambda item: item[1], reverse=True)
    sensitivity_by_layer = {layer: round(drop, 4) for layer, drop in sensitivities}
    locked_layers = derive_locked_layers(sensitivity_by_layer, threshold=LOCK_THRESHOLD)

    save_sensitivity_results(
        {
            "model_name": DEFAULT_MODEL_NAME,
            "num_labels": DEFAULT_NUM_LABELS,
            "bit_simulation": BIT_SIMULATION,
            "baseline_accuracy": round(baseline_acc, 4),
            "lock_threshold": LOCK_THRESHOLD,
            "locked_layers": locked_layers,
            "sensitivity_by_layer": sensitivity_by_layer,
            "sorted_layers": [
                {"layer": layer, "accuracy_drop": round(drop, 4)}
                for layer, drop in sensitivities
            ],
        }
    )

    print("\nMost Sensitive Layers:")
    for layer, drop in sensitivities[:10]:
        print(layer, round(drop, 4))

    print(f"\nLocked layers (threshold {LOCK_THRESHOLD:.4f}):")
    for layer in locked_layers:
        print(layer)


if __name__ == "__main__":
    main()
