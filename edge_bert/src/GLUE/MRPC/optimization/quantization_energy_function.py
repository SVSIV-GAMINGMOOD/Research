from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import load_mrpc_dataset
from GLUE.MRPC.quantization import fake_quantize_tensor
from GLUE.MRPC.task_config import (
    MODEL_NAME,
    NUM_LABELS,
    QUANTIZATION_SETTINGS,
    SEARCH_SETTINGS,
    baseline_checkpoint_path,
    get_locked_layers,
    get_search_reference_metrics,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIT_OPTIONS = QUANTIZATION_SETTINGS["bit_options"]
LOCKED_LAYERS = set(get_locked_layers())
ALPHA = SEARCH_SETTINGS["alpha"]
BETA = SEARCH_SETTINGS["beta"]
GAMMA = SEARCH_SETTINGS["gamma"]
SEARCH_BATCH_SIZE = SEARCH_SETTINGS.get("search_batch_size", 16)

_STATE: dict[str, Any] = {}
_CANDIDATE_CACHE: dict[tuple[tuple[str, int], ...], tuple[float, float, float]] = {}


def _candidate_signature(candidate_bits: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(candidate_bits.items()))


def _build_state(*, verbose: bool = False) -> dict[str, Any]:
    if verbose:
        print("Loading MRPC cached search reference metrics...")
    baseline_size_mb, baseline_latency_ms = get_search_reference_metrics()

    if verbose:
        max_samples = SEARCH_SETTINGS["search_eval_max_samples"]
        sample_label = "full validation split" if max_samples is None else f"{max_samples} validation samples"
        print(f"Loading MRPC search dataset ({sample_label})...")
    val_dataset = load_mrpc_dataset(
        format_type="torch",
        max_samples=SEARCH_SETTINGS["search_eval_max_samples"],
    )

    if verbose:
        print("Loading MRPC baseline checkpoint for search...")
    base_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    base_model.load_state_dict(torch.load(baseline_checkpoint_path(), map_location="cpu"))
    base_model.to(DEVICE)
    base_model.eval()

    searchable_layers: list[str] = []
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name not in LOCKED_LAYERS:
            searchable_layers.append(name)

    if verbose:
        print(
            "MRPC search context ready: "
            f"{len(searchable_layers)} searchable layers, {len(val_dataset)} eval samples, "
            f"batch_size={SEARCH_BATCH_SIZE}, ref_size={baseline_size_mb:.2f}MB, "
            f"ref_latency={baseline_latency_ms:.2f}ms."
        )

    return {
        "baseline_size_mb": baseline_size_mb,
        "baseline_latency_ms": baseline_latency_ms,
        "val_dataset": val_dataset,
        "base_model": base_model,
        "searchable_layers": searchable_layers,
    }


def get_state(*, verbose: bool = False) -> dict[str, Any]:
    global _STATE
    if not _STATE:
        _STATE = _build_state(verbose=verbose)
    elif verbose:
        print(
            "Reusing cached MRPC search context: "
            f"{len(_STATE['searchable_layers'])} searchable layers, {len(_STATE['val_dataset'])} eval samples."
        )
    return _STATE


def warmup_search_context(*, verbose: bool = False) -> dict[str, Any]:
    return get_state(verbose=verbose)


def _get_linear_name(param_name: str) -> str:
    return param_name.replace(".weight", "").replace(".bias", "")


def get_searchable_layers() -> list[str]:
    return list(get_state()["searchable_layers"])


def evaluate(model) -> float:
    predictions: list[int] = []
    labels: list[int] = []
    dataset = get_state()["val_dataset"]
    with torch.no_grad():
        for index in range(0, len(dataset), SEARCH_BATCH_SIZE):
            batch = dataset[index : index + SEARCH_BATCH_SIZE]
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, predictions)


def measure_baseline_accuracy() -> float:
    return evaluate(get_state()["base_model"])


def build_uniform_candidate(bits: int) -> dict[str, int]:
    return {layer: bits for layer in get_searchable_layers()}


def estimate_model_size(candidate_bits: dict[str, int]) -> float:
    base_model = get_state()["base_model"]
    total_bits = 0
    total_params = 0
    for param_name, param in base_model.named_parameters():
        params = param.numel()
        layer_name = _get_linear_name(param_name)
        total_params += params
        if layer_name in LOCKED_LAYERS:
            total_bits += params * 8
        elif layer_name in candidate_bits:
            total_bits += params * candidate_bits[layer_name]
        else:
            total_bits += params * 32
    return total_bits / 8 / 1024 / 1024


def estimate_latency(candidate_bits: dict[str, int]) -> float:
    base_model = get_state()["base_model"]
    baseline_latency_ms = get_state()["baseline_latency_ms"]
    weighted_sum = 0
    total_params = 0
    for param_name, param in base_model.named_parameters():
        layer_name = _get_linear_name(param_name)
        if layer_name in candidate_bits or layer_name in LOCKED_LAYERS:
            params = param.numel()
            bits = 8 if layer_name in LOCKED_LAYERS else candidate_bits.get(layer_name, 8)
            weighted_sum += params * bits
            total_params += params * 8
    if total_params == 0:
        return baseline_latency_ms
    return baseline_latency_ms * (weighted_sum / total_params)


def evaluate_candidate(candidate_bits: dict[str, int]) -> tuple[float, float, float]:
    signature = _candidate_signature(candidate_bits)
    if signature in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[signature]

    base_model = get_state()["base_model"]
    model_copy = copy.deepcopy(base_model)
    for name, module in model_copy.named_modules():
        if isinstance(module, torch.nn.Linear):
            bits = 8 if name in LOCKED_LAYERS else candidate_bits.get(name, 8)
            module.weight.data = fake_quantize_tensor(module.weight.data, bits)

    accuracy = evaluate(model_copy)
    size_mb = estimate_model_size(candidate_bits)
    latency_ms = estimate_latency(candidate_bits)
    result = (accuracy, size_mb, latency_ms)
    _CANDIDATE_CACHE[signature] = result
    return result


def get_size_latency_bounds() -> dict[str, float]:
    min_candidate = build_uniform_candidate(min(BIT_OPTIONS))
    max_candidate = build_uniform_candidate(max(BIT_OPTIONS))
    min_size = estimate_model_size(min_candidate)
    min_latency = estimate_latency(min_candidate)
    max_size = estimate_model_size(max_candidate)
    max_latency = estimate_latency(max_candidate)
    return {
        "size_min": min(min_size, max_size),
        "size_max": max(min_size, max_size),
        "latency_min": min(min_latency, max_latency),
        "latency_max": max(min_latency, max_latency),
    }


def compute_energy(candidate_bits: dict[str, int]) -> tuple[float, float, float, float]:
    accuracy, size_mb, latency_ms = evaluate_candidate(candidate_bits)
    energy = (
        ALPHA * (1 - accuracy)
        + BETA * (latency_ms / get_state()["baseline_latency_ms"])
        + GAMMA * (size_mb / get_state()["baseline_size_mb"])
    )
    return energy, accuracy, size_mb, latency_ms
