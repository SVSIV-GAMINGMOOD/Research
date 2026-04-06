from __future__ import annotations

import argparse
import copy
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from datasets import load_dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

MODEL_ID = "answerdotai/ModernBERT-large"
DATASET_NAME = "glue"
DATASET_CONFIG = "mrpc"
TEXT_COLUMNS = ("sentence1", "sentence2")
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
NUM_LABELS = 2
MAX_LENGTH = 128

BIT_OPTIONS = (4, 6, 8)
DEFAULT_BITS = 32
FP16_BITS = 16
INT8_BITS = 8
HYBRID_EMBED_BITS = 8
SENSITIVITY_SIM_BITS = 4
LOCK_THRESHOLD = 0.01

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_EPOCHS = 5
DEFAULT_EARLY_STOPPING = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_LATENCY_RUNS = 50
DEFAULT_THREADS = 4

SEARCH_THRESHOLD_DROPS = (0.25, 0.5, 0.75, 1.0, 1.25)
SEARCH_INITIAL_TEMPERATURE = 1.0
SEARCH_FINAL_TEMPERATURE = 0.05
SEARCH_COOLING_RATE = 0.9
SEARCH_ITERATIONS_PER_TEMP = 3
SEARCH_EARLY_STOP_PATIENCE = 8
SEARCH_PLATEAU_EPS = 1e-3
SEARCH_NUM_RUNS = 3
SEARCH_DEFAULT_THRESHOLD_INDEX = 0
SEARCH_GUIDE_SIZE_WEIGHT = 1.0
SEARCH_GUIDE_LATENCY_WEIGHT = 1.0
GREEDY_ALLOCATION_STRATEGY = "tertiles"

REQUESTED_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_ORT_PROVIDER = "CPUExecutionProvider"
CUDA_ORT_PROVIDER = "CUDAExecutionProvider"

METHOD_ORDER = [
    "FP32 Research Reference",
    "Primary Baseline (FP16)",
    "INT8 Uniform",
    "Greedy Mixed",
    "SA (v1)",
    "Hybrid SA (ours)",
]

METHOD_COLORS = {
    "FP32 Research Reference": "#607D8B",
    "Primary Baseline (FP16)": "#3F51B5",
    "INT8 Uniform": "#2196F3",
    "Greedy Mixed": "#FF9800",
    "SA (v1)": "#9C27B0",
    "Hybrid SA (ours)": "#F44336",
}

SHORT_LABELS = {
    "FP32 Research Reference": "FP32",
    "Primary Baseline (FP16)": "FP16",
    "INT8 Uniform": "INT8",
    "Greedy Mixed": "Greedy",
    "SA (v1)": "SA v1",
    "Hybrid SA (ours)": "Hybrid SA",
}


@dataclass(slots=True)
class OutputLayout:
    root: Path
    models_dir: Path
    results_dir: Path
    figures_dir: Path


@dataclass(slots=True)
class TrainingSpec:
    label: str
    checkpoint_name: str
    use_amp_training: bool
    metric_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModernBERT-large MRPC Kaggle pipeline")
    parser.add_argument("--phase", choices=["all", "train", "search", "export", "evaluate", "figures"], default="all")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "modernbert_large_kaggle")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--latency-runs", type=int, default=DEFAULT_LATENCY_RUNS)
    parser.add_argument("--keep-search-history", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args, _ = parser.parse_known_args()
    return args


def build_layout(root: Path) -> OutputLayout:
    root = root.resolve()
    models_dir = root / "models"
    results_dir = root / "results"
    figures_dir = root / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return OutputLayout(root=root, models_dir=models_dir, results_dir=results_dir, figures_dir=figures_dir)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(*, require_cuda: bool = False) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this phase, but no CUDA device is available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(layout: OutputLayout, checkpoint_name: str) -> Path:
    return layout.models_dir / checkpoint_name


def onnx_path(layout: OutputLayout, file_name: str) -> Path:
    return layout.models_dir / file_name


def results_path(layout: OutputLayout, file_name: str) -> Path:
    return layout.results_dir / file_name


def figures_path(layout: OutputLayout, file_name: str) -> Path:
    return layout.figures_dir / file_name


def build_tokenizer() -> AutoTokenizer:
    kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(MODEL_ID)


def _attempt_model_load(extra_kwargs: dict[str, Any]) -> Any:
    kwargs = {
        "num_labels": NUM_LABELS,
        "ignore_mismatched_sizes": True,
        "trust_remote_code": True,
    }
    kwargs.update(extra_kwargs)
    try:
        return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **kwargs)


def build_model() -> Any:
    model = None
    last_error = None
    for attempt in ({"attn_implementation": "eager"}, {}):
        try:
            model = _attempt_model_load(attempt)
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
    if model is None:
        raise RuntimeError(f"Unable to load {MODEL_ID}: {last_error}")

    if hasattr(model, "config"):
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    return model


def load_mrpc_dataset(
    tokenizer: Any,
    *,
    split: str,
    format_type: str,
    max_samples: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = DEFAULT_SEED,
) -> Any:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[TEXT_COLUMNS[0]],
            batch[TEXT_COLUMNS[1]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize, batched=True)
    label_column = "label"
    if format_type == "torch":
        dataset = dataset.rename_column("label", "labels")
        label_column = "labels"
    dataset.set_format(type=format_type, columns=["input_ids", "attention_mask", label_column])
    return dataset


def create_training_dataloaders(
    tokenizer: Any,
    *,
    batch_size: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader, int]:
    train_dataset = load_mrpc_dataset(
        tokenizer,
        split=TRAIN_SPLIT,
        format_type="torch",
        max_samples=max_train_samples,
        shuffle=True,
        shuffle_seed=seed,
    )
    validation_dataset = load_mrpc_dataset(
        tokenizer,
        split=VALIDATION_SPLIT,
        format_type="torch",
        max_samples=max_eval_samples,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size)
    return train_loader, val_loader, len(validation_dataset)


def load_model_from_checkpoint(checkpoint_file: Path, *, device: torch.device, use_half: bool = False) -> Any:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")
    model = build_model()
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    if use_half:
        model = model.half()
    model.eval()
    return model


def softmax_np(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_pytorch_model(
    model: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    latency_runs: int,
    use_amp_inference: bool,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[int] = []
    labels: list[int] = []
    probs: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions.extend(torch.argmax(probabilities, dim=-1).cpu().tolist())
            probs.extend(probabilities[:, 1].cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    sample = dataset[0]
    example_inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }
    for _ in range(10):
        with torch.no_grad():
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                model(**example_inputs)
    latencies = []
    for _ in range(latency_runs):
        _sync_if_needed(device)
        start = time.perf_counter()
        with torch.no_grad():
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                model(**example_inputs)
        _sync_if_needed(device)
        latencies.append((time.perf_counter() - start) * 1000)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = -1.0

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc,
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_validation_loader(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp_inference: bool,
) -> tuple[float, float]:
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                logits = model(**inputs).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, predictions), f1_score(labels, predictions)


# DATASET_AND_MODEL_HELPERS
def train_single_baseline(
    layout: OutputLayout,
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    spec: TrainingSpec,
) -> dict[str, Any]:
    seed_everything(args.seed)
    device = get_device(require_cuda=spec.use_amp_training)
    model = build_model().to(device)

    train_loader, val_loader, validation_size = create_training_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )
    steps_per_epoch = len(train_loader)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max(1, args.epochs * max(1, steps_per_epoch)),
    )
    scaler = GradScaler(enabled=spec.use_amp_training and device.type == "cuda")

    checkpoint_file = checkpoint_path(layout, spec.checkpoint_name)
    best_accuracy = float("-inf")
    best_f1 = 0.0
    best_epoch = 0
    best_step = 0
    global_step = 0
    early_stop_counter = 0
    history: list[dict[str, Any]] = []
    stopped_early = False

    print(f"Training {spec.label} on {device} | amp={spec.use_amp_training}")
    for epoch_index in range(args.epochs):
        epoch_number = epoch_index + 1
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"{spec.label} epoch {epoch_number}/{args.epochs}")
        for batch in progress:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=spec.use_amp_training and device.type == "cuda"):
                outputs = model(**inputs)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            global_step += 1

        train_loss = total_loss / max(1, steps_per_epoch)
        accuracy, f1 = evaluate_validation_loader(
            model,
            val_loader,
            device=device,
            use_amp_inference=spec.use_amp_training and device.type == "cuda",
        )
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            best_f1 = f1
            best_epoch = epoch_number
            best_step = global_step
            torch.save(model.state_dict(), checkpoint_file)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        history.append(
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "train_loss": round(train_loss, 4),
                "validation_accuracy": round(accuracy, 4),
                "validation_f1": round(f1, 4),
                "is_best_checkpoint": is_best,
            }
        )
        print(
            f"{spec.label} epoch {epoch_number}: "
            f"loss={train_loss:.4f} acc={accuracy:.4f} f1={f1:.4f} best={best_accuracy:.4f}"
        )
        if early_stop_counter >= DEFAULT_EARLY_STOPPING:
            stopped_early = True
            print(f"{spec.label} early stopping triggered at epoch {epoch_number}.")
            break

    return {
        "label": spec.label,
        "checkpoint": checkpoint_file.name,
        "checkpoint_path": str(checkpoint_file.relative_to(layout.root)).replace("\\", "/"),
        "training_precision": spec.metric_dtype,
        "used_amp_training": spec.use_amp_training,
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "steps_per_epoch": steps_per_epoch,
        "best_epoch": best_epoch,
        "best_global_step": best_step,
        "best_validation_accuracy": round(best_accuracy, 4),
        "best_validation_f1": round(best_f1, 4),
        "stopped_early": stopped_early,
        "validation_samples": validation_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "history": history,
    }


def run_training_phase(layout: OutputLayout, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = build_tokenizer()
    specs = [
        TrainingSpec(
            label="FP32 Research Reference",
            checkpoint_name="fp32_reference_best.pt",
            use_amp_training=False,
            metric_dtype="fp32",
        ),
        TrainingSpec(
            label="Primary Baseline (FP16)",
            checkpoint_name="fp16_baseline_best.pt",
            use_amp_training=True,
            metric_dtype="fp16",
        ),
    ]

    runs: dict[str, Any] = {}
    for spec in specs:
        runs[spec.label] = train_single_baseline(layout, args, tokenizer, spec=spec)

    payload = {
        "model_id": MODEL_ID,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "text_columns": list(TEXT_COLUMNS),
        "max_length": MAX_LENGTH,
        "runs": runs,
    }
    save_json(results_path(layout, "training_history.json"), payload)
    return payload


# TRAINING_PHASE
def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = 0.0
    qmax = (2.0**bits) - 1.0
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    q_tensor = zero_point + tensor / (scale + 1e-8)
    q_tensor.clamp_(qmin, qmax).round_()
    return scale * (q_tensor - zero_point)


def evaluate_model_accuracy(
    model: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, predictions)


def build_search_dataset(tokenizer: Any, max_eval_samples: int | None) -> Any:
    return load_mrpc_dataset(
        tokenizer,
        split=VALIDATION_SPLIT,
        format_type="torch",
        max_samples=max_eval_samples,
    )


def derive_locked_layers(sensitivity_by_layer: dict[str, float], threshold: float = LOCK_THRESHOLD) -> list[str]:
    return sorted(layer for layer, drop in sensitivity_by_layer.items() if drop >= threshold)


def run_sensitivity_analysis(
    layout: OutputLayout,
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    device = get_device()
    fp32_checkpoint = checkpoint_path(layout, "fp32_reference_best.pt")
    if not fp32_checkpoint.exists():
        raise FileNotFoundError("Train the FP32 reference first before running search.")

    dataset = build_search_dataset(tokenizer, args.max_eval_samples)
    base_model = load_model_from_checkpoint(fp32_checkpoint, device=device, use_half=False)
    baseline_accuracy = evaluate_model_accuracy(base_model, dataset, device=device, batch_size=args.batch_size)

    sensitivities = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            model_copy = copy.deepcopy(base_model)
            target_layer = dict(model_copy.named_modules())[name]
            target_layer.weight.data = fake_quantize_tensor(target_layer.weight.data, SENSITIVITY_SIM_BITS)
            accuracy = evaluate_model_accuracy(model_copy, dataset, device=device, batch_size=args.batch_size)
            sensitivities.append((name, baseline_accuracy - accuracy))
            print(f"Sensitivity {name}: drop={baseline_accuracy - accuracy:.4f}")

    sensitivities.sort(key=lambda item: item[1], reverse=True)
    sensitivity_by_layer = {layer: round(drop, 4) for layer, drop in sensitivities}
    locked_layers = derive_locked_layers(sensitivity_by_layer, LOCK_THRESHOLD)
    payload = {
        "model_id": MODEL_ID,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "reference_checkpoint": fp32_checkpoint.name,
        "bit_simulation": SENSITIVITY_SIM_BITS,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "lock_threshold": LOCK_THRESHOLD,
        "max_eval_samples": args.max_eval_samples,
        "locked_layers": locked_layers,
        "sensitivity_by_layer": sensitivity_by_layer,
        "sorted_layers": [
            {"layer": layer, "accuracy_drop": round(drop, 4)}
            for layer, drop in sensitivities
        ],
    }
    save_json(results_path(layout, "sensitivity_results.json"), payload)
    return payload


def generate_greedy_config(
    layout: OutputLayout,
    sensitivity_results: dict[str, Any],
) -> dict[str, int]:
    sensitivity = sensitivity_results["sensitivity_by_layer"]
    locked_layers = set(sensitivity_results["locked_layers"])
    search_layers = {layer: drop for layer, drop in sensitivity.items() if layer not in locked_layers}
    sorted_layers = sorted(search_layers.items(), key=lambda item: item[1], reverse=True)

    if GREEDY_ALLOCATION_STRATEGY != "tertiles":
        raise ValueError(f"Unsupported greedy allocation strategy: {GREEDY_ALLOCATION_STRATEGY}")

    count = len(sorted_layers)
    top = sorted_layers[: count // 3]
    middle = sorted_layers[count // 3 : (2 * count) // 3]
    low = sorted_layers[(2 * count) // 3 :]
    config: dict[str, int] = {}
    for layer, _ in top:
        config[layer] = 8
    for layer, _ in middle:
        config[layer] = 6
    for layer, _ in low:
        config[layer] = 4
    for layer in locked_layers:
        config[layer] = 8

    save_json(results_path(layout, "greedy_config.json"), config)
    return config


def get_linear_module_names(model: Any, locked_layers: set[str]) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and name not in locked_layers]


def compute_theoretical_size_mb(
    model: Any,
    *,
    linear_config: dict[str, int] | None = None,
    embed_bits: int = DEFAULT_BITS,
    default_bits: int = DEFAULT_BITS,
) -> dict[str, float]:
    embedding_prefixes = [name for name, module in model.named_modules() if isinstance(module, nn.Embedding)]
    total_bits = 0
    total_params = 0
    for param_name, param in model.named_parameters():
        total_params += param.numel()
        base_name = param_name.rsplit(".", 1)[0]
        if any(base_name == prefix or base_name.startswith(prefix + ".") for prefix in embedding_prefixes):
            total_bits += param.numel() * embed_bits
        elif linear_config and base_name in linear_config:
            total_bits += param.numel() * linear_config[base_name]
        else:
            total_bits += param.numel() * default_bits
    return {
        "theoretical_mb": total_bits / 8 / 1024 / 1024,
        "effective_bits": total_bits / max(1, total_params),
        "total_params": total_params,
    }


def measure_reference_latency_ms(
    model: Any,
    dataset: Any,
    *,
    device: torch.device,
    use_amp_inference: bool,
) -> float:
    sample = dataset[0]
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }
    for _ in range(10):
        with torch.no_grad():
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                model(**inputs)
    latencies = []
    for _ in range(max(20, DEFAULT_LATENCY_RUNS)):
        _sync_if_needed(device)
        start = time.perf_counter()
        with torch.no_grad():
            with autocast(enabled=use_amp_inference and device.type == "cuda"):
                model(**inputs)
        _sync_if_needed(device)
        latencies.append((time.perf_counter() - start) * 1000)
    return float(np.mean(latencies))


def estimate_candidate_latency_ms(
    model: Any,
    candidate_bits: dict[str, int],
    *,
    locked_layers: set[str],
    baseline_latency_ms: float,
) -> float:
    weighted_sum = 0
    total_params = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        params = module.weight.numel()
        bits = 8 if name in locked_layers else candidate_bits.get(name, 8)
        weighted_sum += params * bits
        total_params += params * 8
    if total_params == 0:
        return baseline_latency_ms
    return baseline_latency_ms * (weighted_sum / total_params)


def build_uniform_candidate(searchable_layers: list[str], bits: int) -> dict[str, int]:
    return {layer: bits for layer in searchable_layers}


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def candidate_signature(candidate: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(candidate.items()))


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    better_on_one = a["size_mb"] < b["size_mb"] or a["latency_ms"] < b["latency_ms"]
    no_worse = a["size_mb"] <= b["size_mb"] and a["latency_ms"] <= b["latency_ms"]
    return better_on_one and no_worse


def insert_frontier_candidate(frontier: list[dict[str, Any]], candidate_record: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate_record["feasible"]:
        return frontier

    signature = candidate_signature(candidate_record["candidate"])
    for existing in frontier:
        existing_signature = candidate_signature(existing["candidate"])
        if existing_signature == signature and existing["guide_energy"] <= candidate_record["guide_energy"]:
            return frontier
        if dominates(existing, candidate_record):
            return frontier

    pruned = [
        existing
        for existing in frontier
        if candidate_signature(existing["candidate"]) != signature and not dominates(candidate_record, existing)
    ]
    pruned.append(candidate_record)
    return sorted(pruned, key=lambda item: (item["size_mb"], item["latency_ms"], -item["accuracy"]))


def merge_frontiers(frontiers: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for frontier in frontiers:
        for candidate in frontier:
            merged = insert_frontier_candidate(merged, candidate)
    return merged


def run_sa_search(
    layout: OutputLayout,
    args: argparse.Namespace,
    tokenizer: Any,
    sensitivity_results: dict[str, Any],
) -> dict[str, Any]:
    device = get_device()
    fp32_checkpoint = checkpoint_path(layout, "fp32_reference_best.pt")
    base_model = load_model_from_checkpoint(fp32_checkpoint, device=device, use_half=False)
    dataset = build_search_dataset(tokenizer, args.max_eval_samples)
    locked_layers = set(sensitivity_results["locked_layers"])
    searchable_layers = get_linear_module_names(base_model, locked_layers)
    baseline_accuracy = evaluate_model_accuracy(base_model, dataset, device=device, batch_size=args.batch_size)
    baseline_latency_ms = measure_reference_latency_ms(base_model, dataset, device=device, use_amp_inference=False)
    baseline_size_mb = compute_theoretical_size_mb(base_model)["theoretical_mb"]

    cache: dict[tuple[tuple[str, int], ...], tuple[float, float, float]] = {}

    def evaluate_candidate(candidate_bits: dict[str, int]) -> tuple[float, float, float]:
        signature = candidate_signature(candidate_bits)
        if signature in cache:
            return cache[signature]

        model_copy = copy.deepcopy(base_model)
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear):
                bits = 8 if name in locked_layers else candidate_bits.get(name, 8)
                module.weight.data = fake_quantize_tensor(module.weight.data, bits)
        accuracy = evaluate_model_accuracy(model_copy, dataset, device=device, batch_size=args.batch_size)
        size_payload = compute_theoretical_size_mb(
            model_copy,
            linear_config={name: 8 if name in locked_layers else candidate_bits.get(name, 8) for name, module in base_model.named_modules() if isinstance(module, nn.Linear)},
        )
        latency = estimate_candidate_latency_ms(
            base_model,
            candidate_bits,
            locked_layers=locked_layers,
            baseline_latency_ms=baseline_latency_ms,
        )
        result = (accuracy, size_payload["theoretical_mb"], latency)
        cache[signature] = result
        return result

    def get_bounds() -> dict[str, float]:
        minimum = build_uniform_candidate(searchable_layers, min(BIT_OPTIONS))
        maximum = build_uniform_candidate(searchable_layers, max(BIT_OPTIONS))
        _, min_size, min_latency = evaluate_candidate(minimum)
        _, max_size, max_latency = evaluate_candidate(maximum)
        return {
            "size_min": min(min_size, max_size),
            "size_max": max(min_size, max_size),
            "latency_min": min(min_latency, max_latency),
            "latency_max": max(min_latency, max_latency),
        }

    def evaluate_candidate_record(
        candidate: dict[str, int],
        threshold_spec: dict[str, Any],
        bounds: dict[str, float],
        seed: int,
    ) -> dict[str, Any]:
        accuracy, size_mb, latency_ms = evaluate_candidate(candidate)
        feasible = accuracy >= float(threshold_spec["accuracy_threshold"])
        guide_energy = float("inf")
        if feasible:
            guide_energy = (
                SEARCH_GUIDE_SIZE_WEIGHT * normalize(size_mb, bounds["size_min"], bounds["size_max"])
                + SEARCH_GUIDE_LATENCY_WEIGHT * normalize(latency_ms, bounds["latency_min"], bounds["latency_max"])
            )
        return {
            "seed": seed,
            "threshold_label": threshold_spec["label"],
            "threshold_drop_percent": threshold_spec["drop_percent"],
            "threshold_accuracy": float(threshold_spec["accuracy_threshold"]),
            "accuracy": accuracy,
            "size_mb": size_mb,
            "latency_ms": latency_ms,
            "feasible": feasible,
            "guide_energy": guide_energy,
            "candidate": copy.deepcopy(candidate),
        }

    def run_single_search(seed: int, threshold_spec: dict[str, Any], bounds: dict[str, float]) -> dict[str, Any]:
        random.seed(seed)
        np.random.seed(seed)
        current_candidate = build_uniform_candidate(searchable_layers, max(BIT_OPTIONS))
        current_record = evaluate_candidate_record(current_candidate, threshold_spec, bounds, seed)
        if not current_record["feasible"]:
            return {
                "seed": seed,
                "threshold_label": threshold_spec["label"],
                "threshold_drop_percent": threshold_spec["drop_percent"],
                "threshold_accuracy": threshold_spec["accuracy_threshold"],
                "history": [],
                "best_result": None,
                "frontier": [],
                "feasible_start": False,
            }

        best_record = copy.deepcopy(current_record)
        frontier = insert_frontier_candidate([], current_record)
        temperature = SEARCH_INITIAL_TEMPERATURE
        no_improve_counter = 0
        history: list[float] = []

        while temperature > SEARCH_FINAL_TEMPERATURE:
            previous_best_energy = best_record["guide_energy"]
            for _ in range(SEARCH_ITERATIONS_PER_TEMP):
                new_candidate = dict(current_candidate)
                layer_to_modify = random.choice(searchable_layers)
                current_bit = new_candidate[layer_to_modify]
                new_candidate[layer_to_modify] = random.choice([bit for bit in BIT_OPTIONS if bit != current_bit])
                new_record = evaluate_candidate_record(new_candidate, threshold_spec, bounds, seed)
                if not new_record["feasible"]:
                    continue

                frontier = insert_frontier_candidate(frontier, new_record)
                delta = new_record["guide_energy"] - current_record["guide_energy"]
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-8)):
                    current_candidate = dict(new_candidate)
                    current_record = new_record
                if new_record["guide_energy"] < best_record["guide_energy"]:
                    best_record = copy.deepcopy(new_record)

            history.append(best_record["guide_energy"])
            improvement = previous_best_energy - best_record["guide_energy"]
            if improvement < SEARCH_PLATEAU_EPS:
                no_improve_counter += 1
            else:
                no_improve_counter = 0
            if no_improve_counter >= SEARCH_EARLY_STOP_PATIENCE:
                break
            temperature *= SEARCH_COOLING_RATE

        return {
            "seed": seed,
            "threshold_label": threshold_spec["label"],
            "threshold_drop_percent": threshold_spec["drop_percent"],
            "threshold_accuracy": threshold_spec["accuracy_threshold"],
            "history": history,
            "best_result": best_record,
            "frontier": frontier,
            "feasible_start": True,
        }

    threshold_specs = [
        {
            "label": f"baseline_minus_{str(drop).replace('.', 'p')}pct",
            "drop_percent": float(drop),
            "accuracy_threshold": max(0.0, baseline_accuracy - (float(drop) / 100.0)),
        }
        for drop in SEARCH_THRESHOLD_DROPS
    ]
    bounds = get_bounds()
    threshold_results = []
    for threshold_spec in threshold_specs:
        runs = [run_single_search(seed, threshold_spec, bounds) for seed in range(SEARCH_NUM_RUNS)]
        merged_frontier = merge_frontiers([run["frontier"] for run in runs])
        representative = min(merged_frontier, key=lambda item: item["guide_energy"]) if merged_frontier else None
        threshold_results.append(
            {
                "label": threshold_spec["label"],
                "drop_percent": threshold_spec["drop_percent"],
                "accuracy_threshold": threshold_spec["accuracy_threshold"],
                "runs": runs,
                "frontier": merged_frontier,
                "representative_result": representative,
            }
        )

    default_result = None
    feasible_thresholds = [result for result in threshold_results if result.get("representative_result")]
    if feasible_thresholds:
        index = max(0, min(SEARCH_DEFAULT_THRESHOLD_INDEX, len(threshold_results) - 1))
        for candidate_index in range(index, len(threshold_results)):
            representative = threshold_results[candidate_index].get("representative_result")
            if representative:
                default_result = representative
                break
        if default_result is None:
            default_result = feasible_thresholds[0]["representative_result"]

    if default_result is None:
        raise RuntimeError("No feasible SA configuration found.")

    payload = {
        "model_id": MODEL_ID,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "reference_checkpoint": fp32_checkpoint.name,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "baseline_latency_ms": round(baseline_latency_ms, 4),
        "baseline_size_mb": round(baseline_size_mb, 4),
        "locked_layers": sorted(locked_layers),
        "searchable_layer_count": len(searchable_layers),
        "default_result": {
            "threshold_label": default_result["threshold_label"],
            "threshold_accuracy": round(default_result["threshold_accuracy"], 4),
            "threshold_drop_percent": default_result["threshold_drop_percent"],
            "seed": default_result["seed"],
            "accuracy": round(default_result["accuracy"], 4),
            "size_mb": round(default_result["size_mb"], 4),
            "latency_ms": round(default_result["latency_ms"], 4),
            "guide_energy": round(default_result["guide_energy"], 6),
            "bit_config": default_result["candidate"],
        },
    }
    save_json(results_path(layout, "sa_best_config.json"), payload)

    if args.keep_search_history:
        summary = {
            "baseline_accuracy": payload["baseline_accuracy"],
            "baseline_latency_ms": payload["baseline_latency_ms"],
            "baseline_size_mb": payload["baseline_size_mb"],
            "thresholds": threshold_results,
            "default_result": payload["default_result"],
        }
        save_json(results_path(layout, "sa_search_summary.json"), summary)

    return payload


def run_search_phase(layout: OutputLayout, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = build_tokenizer()
    sensitivity_results = run_sensitivity_analysis(layout, args, tokenizer)
    greedy_config = generate_greedy_config(layout, sensitivity_results)
    sa_best_config = run_sa_search(layout, args, tokenizer, sensitivity_results)
    return {
        "sensitivity_results": sensitivity_results,
        "greedy_config": greedy_config,
        "sa_best_config": sa_best_config,
    }


# SEARCH_PHASE
def cleanup_onnx_artifacts(model_path: Path) -> None:
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if model_path.exists():
        model_path.unlink()
    if sidecar.exists():
        sidecar.unlink()


def consolidate_onnx_to_single_file(model_path: Path) -> None:
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if not sidecar.exists():
        return
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.save_model(model, str(model_path), save_as_external_data=False)
    if sidecar.exists():
        sidecar.unlink()


def total_onnx_size_mb(model_path: Path) -> float:
    size = 0
    if model_path.exists():
        size += model_path.stat().st_size
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if sidecar.exists():
        size += sidecar.stat().st_size
    return size / (1024**2)


class SequenceClassifierExportWrapper(nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear_layer.weight.data
        q_min = 0
        q_max = (2**bits) - 1
        w_min = weight.min()
        w_max = weight.max()
        scale = (w_max - w_min) / (q_max - q_min + 1e-8)
        zero_point = torch.round(q_min - (w_min / (scale + 1e-8)))
        q_weight = torch.clamp(torch.round(weight / (scale + 1e-8) + zero_point), q_min, q_max).to(torch.int32)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return nn.functional.linear(x, weight, self.bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding):
        super().__init__()
        weight = embedding_layer.weight.data
        q_min, q_max = -128, 127
        abs_max = weight.abs().max(dim=1, keepdim=True).values + 1e-8
        scale = abs_max / 127.0
        q_weight = torch.clamp(torch.round(weight / scale), q_min, q_max).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.num_embeddings = embedding_layer.num_embeddings
        self.embedding_dim = embedding_layer.embedding_dim
        self.padding_idx = embedding_layer.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        quant_rows = self.q_weight[input_ids]
        scales = self.scale[input_ids]
        return quant_rows.float() * scales


def replace_linear_layers(model: Any, config: dict[str, int]) -> Any:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedLinear(module, config.get(name, 8)))
    return model


def replace_embedding_layers(model: Any) -> Any:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Embedding):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedEmbedding(module))
    return model


def export_model_to_onnx(
    model: Any,
    output_path: Path,
    *,
    tokenizer: Any,
    use_half: bool,
) -> dict[str, Any]:
    cleanup_onnx_artifacts(output_path)
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_half and requested_device.type != "cuda":
        raise RuntimeError("FP16 ONNX export requires CUDA.")

    fallback_reason = None
    dummy = tokenizer(
        "Sentence A",
        "Sentence B",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    device_attempts = [requested_device]
    if requested_device.type == "cuda" and not use_half:
        device_attempts.append(torch.device("cpu"))

    for target_device in device_attempts:
        try:
            model = model.to(target_device)
            if use_half:
                model = model.half()
            else:
                model = model.float()
            model.eval()
            wrapped = SequenceClassifierExportWrapper(model).to(target_device).eval()
            dummy_ids = dummy["input_ids"].to(target_device)
            dummy_mask = dummy["attention_mask"].to(target_device)
            with torch.no_grad():
                export_kwargs = {
                    "input_names": ["input_ids", "attention_mask"],
                    "output_names": ["logits"],
                    "dynamic_axes": {
                        "input_ids": {0: "batch_size"},
                        "attention_mask": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                    "opset_version": 18,
                    "do_constant_folding": False,
                    "export_params": True,
                }
                if use_half:
                    export_kwargs["dynamo"] = False
                    export_kwargs["external_data"] = False
                torch.onnx.export(
                    wrapped,
                    (dummy_ids, dummy_mask),
                    str(output_path),
                    **export_kwargs,
                )
            consolidate_onnx_to_single_file(output_path)
            return {"export_device": str(target_device), "fallback_reason": fallback_reason}
        except Exception as exc:  # pragma: no cover - runtime specific
            if target_device.type == "cpu" or use_half:
                raise
            fallback_reason = f"CUDA ONNX export failed; retried on CPU: {exc}"

    raise RuntimeError(f"Failed to export ONNX: {output_path}")


def sanitize_and_quantize_dynamic(fp32_model_path: Path, int8_model_path: Path) -> None:
    cleanup_onnx_artifacts(int8_model_path)
    with tempfile.TemporaryDirectory(prefix="modernbert.quant.") as tmp_dir:
        sanitized_fp32_path = Path(tmp_dir) / "sanitized_fp32.onnx"
        model = onnx.load(str(fp32_model_path))
        del model.graph.value_info[:]

        input_batch_param = None
        for inp in model.graph.input:
            tensor_type = inp.type.tensor_type
            if tensor_type.HasField("shape") and len(tensor_type.shape.dim) > 0:
                dim = tensor_type.shape.dim[0]
                if dim.dim_param:
                    input_batch_param = dim.dim_param
                    break

        if input_batch_param:
            for out in model.graph.output:
                tensor_type = out.type.tensor_type
                if tensor_type.HasField("shape") and len(tensor_type.shape.dim) > 0:
                    dim = tensor_type.shape.dim[0]
                    if dim.dim_value == 1 and not dim.dim_param:
                        dim.dim_param = input_batch_param
                        dim.dim_value = 0

        onnx.save_model(model, str(sanitized_fp32_path))
        quantize_dynamic(
            model_input=str(sanitized_fp32_path),
            model_output=str(int8_model_path),
            weight_type=QuantType.QInt8,
        )
    consolidate_onnx_to_single_file(int8_model_path)


def build_hybrid_linear_config(layout: OutputLayout) -> dict[str, int]:
    sa_payload = load_json(results_path(layout, "sa_best_config.json"))
    sensitivity_payload = load_json(results_path(layout, "sensitivity_results.json"), default={}) or {}
    if not sa_payload:
        raise FileNotFoundError("Missing sa_best_config.json. Run the search phase first.")
    config = dict(sa_payload["default_result"]["bit_config"])
    for layer in sensitivity_payload.get("locked_layers", []):
        config[layer] = 8
    return config


def export_quantized_variant(
    layout: OutputLayout,
    tokenizer: Any,
    *,
    checkpoint_file: Path,
    output_file_name: str,
    linear_config: dict[str, int],
    quantize_embeddings: bool,
) -> dict[str, Any]:
    device = get_device()
    model = load_model_from_checkpoint(checkpoint_file, device=device, use_half=False)
    if quantize_embeddings:
        model = replace_embedding_layers(model)
    model = replace_linear_layers(model, linear_config)
    return export_model_to_onnx(model, onnx_path(layout, output_file_name), tokenizer=tokenizer, use_half=False)


def run_export_phase(layout: OutputLayout, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = build_tokenizer()
    fp32_checkpoint = checkpoint_path(layout, "fp32_reference_best.pt")
    fp16_checkpoint = checkpoint_path(layout, "fp16_baseline_best.pt")
    if not fp32_checkpoint.exists() or not fp16_checkpoint.exists():
        raise FileNotFoundError("Train both FP32 and FP16 baselines before export.")

    export_metadata: dict[str, Any] = {}

    fp32_model = load_model_from_checkpoint(fp32_checkpoint, device=get_device(), use_half=False)
    export_metadata["FP32 Research Reference"] = export_model_to_onnx(
        fp32_model,
        onnx_path(layout, "fp32_reference.onnx"),
        tokenizer=tokenizer,
        use_half=False,
    )

    fp16_model = load_model_from_checkpoint(fp16_checkpoint, device=get_device(require_cuda=True), use_half=False)
    export_metadata["Primary Baseline (FP16)"] = export_model_to_onnx(
        fp16_model,
        onnx_path(layout, "primary_baseline_fp16.onnx"),
        tokenizer=tokenizer,
        use_half=True,
    )

    sanitize_and_quantize_dynamic(
        onnx_path(layout, "fp32_reference.onnx"),
        onnx_path(layout, "int8_uniform.onnx"),
    )
    export_metadata["INT8 Uniform"] = {
        "export_device": "onnxruntime.quantization",
        "fallback_reason": None,
    }

    greedy_config = load_json(results_path(layout, "greedy_config.json"))
    if not greedy_config:
        raise FileNotFoundError("Missing greedy_config.json. Run the search phase first.")
    sa_linear_config = build_hybrid_linear_config(layout)
    sensitivity_payload = load_json(results_path(layout, "sensitivity_results.json"), default={}) or {}

    export_metadata["Greedy Mixed"] = export_quantized_variant(
        layout,
        tokenizer,
        checkpoint_file=fp32_checkpoint,
        output_file_name="greedy_mixed.onnx",
        linear_config=greedy_config,
        quantize_embeddings=False,
    )
    export_metadata["SA (v1)"] = export_quantized_variant(
        layout,
        tokenizer,
        checkpoint_file=fp32_checkpoint,
        output_file_name="sa_v1.onnx",
        linear_config=sa_linear_config,
        quantize_embeddings=False,
    )
    export_metadata["Hybrid SA (ours)"] = export_quantized_variant(
        layout,
        tokenizer,
        checkpoint_file=fp32_checkpoint,
        output_file_name="hybrid_sa.onnx",
        linear_config=sa_linear_config,
        quantize_embeddings=True,
    )

    fp32_reference_model = load_model_from_checkpoint(fp32_checkpoint, device=torch.device("cpu"), use_half=False)
    hybrid_size = compute_theoretical_size_mb(
        fp32_reference_model,
        linear_config=sa_linear_config,
        embed_bits=HYBRID_EMBED_BITS,
        default_bits=DEFAULT_BITS,
    )
    hybrid_summary = {
        "model_id": MODEL_ID,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "hybrid_sa_config": sa_linear_config,
        "locked_layers": sensitivity_payload.get("locked_layers", []),
        "embedding_bits": HYBRID_EMBED_BITS,
        "theoretical_mb": round(hybrid_size["theoretical_mb"], 4),
        "effective_bits": round(hybrid_size["effective_bits"], 4),
        "onnx_mb": round(total_onnx_size_mb(onnx_path(layout, "hybrid_sa.onnx")), 4),
        "compression_ratio": round(
            compute_theoretical_size_mb(fp32_reference_model)["theoretical_mb"] / hybrid_size["theoretical_mb"],
            4,
        ),
        "exports": export_metadata,
    }
    save_json(results_path(layout, "hybrid_config_summary.json"), hybrid_summary)
    generate_size_comparison(layout)
    return export_metadata


# EXPORT_PHASE
def create_ort_session(
    model_path: Path,
    *,
    requested_providers: list[str],
    threads: int,
) -> tuple[ort.InferenceSession, dict[str, Any]]:
    available = ort.get_available_providers()
    usable = [provider for provider in requested_providers if provider in available]
    fallback_reason = None
    if CUDA_ORT_PROVIDER not in usable and CPU_ORT_PROVIDER in available:
        fallback_reason = f"{CUDA_ORT_PROVIDER} unavailable. Available providers: {available}"
    if not usable:
        usable = [CPU_ORT_PROVIDER] if CPU_ORT_PROVIDER in available else available
        fallback_reason = f"No requested providers available. Available providers: {available}"

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=usable)
    metadata = {
        "requested_provider": requested_providers[0],
        "requested_providers": requested_providers,
        "active_providers": session.get_providers(),
        "execution_backend": "onnxruntime_cuda" if session.get_providers() and session.get_providers()[0] == CUDA_ORT_PROVIDER else "onnxruntime_cpu",
        "fallback_reason": fallback_reason,
    }
    return session, metadata


def run_onnx_inference(
    session: ort.InferenceSession,
    dataset: Any,
    *,
    batch_size: int,
) -> tuple[list[int], list[int], list[float]]:
    predictions: list[int] = []
    labels: list[int] = []
    probs: list[float] = []
    for index in range(0, len(dataset), batch_size):
        batch = dataset[index : index + batch_size]
        inputs = {
            "input_ids": batch["input_ids"].astype(np.int64),
            "attention_mask": batch["attention_mask"].astype(np.int64),
        }
        logits = session.run(["logits"], inputs)[0]
        probabilities = softmax_np(logits.astype(np.float32))
        predictions.extend(np.argmax(probabilities, axis=-1).tolist())
        probs.extend(probabilities[:, 1].tolist())
        labels.extend(batch["label"].tolist())
    return predictions, labels, probs


def evaluate_onnx_model(
    model_path: Path,
    dataset: Any,
    *,
    batch_size: int,
    latency_runs: int,
    threads: int,
) -> dict[str, Any]:
    session, metadata = create_ort_session(model_path, requested_providers=REQUESTED_ORT_PROVIDERS, threads=threads)

    try:
        predictions, labels, probs = run_onnx_inference(session, dataset, batch_size=batch_size)
    except Exception as exc:  # pragma: no cover - runtime specific
        if CPU_ORT_PROVIDER not in ort.get_available_providers():
            raise
        session = ort.InferenceSession(str(model_path), providers=[CPU_ORT_PROVIDER])
        metadata["active_providers"] = session.get_providers()
        metadata["execution_backend"] = "onnxruntime_cpu"
        metadata["fallback_reason"] = f"CPU fallback after provider failure: {exc}"
        predictions, labels, probs = run_onnx_inference(session, dataset, batch_size=batch_size)

    sample = dataset[0]
    example_inputs = {
        "input_ids": sample["input_ids"][np.newaxis].astype(np.int64),
        "attention_mask": sample["attention_mask"][np.newaxis].astype(np.int64),
    }
    for _ in range(10):
        session.run(["logits"], example_inputs)
    latencies = []
    for _ in range(latency_runs):
        start = time.perf_counter()
        session.run(["logits"], example_inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = -1.0

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc,
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "onnx_size_mb": total_onnx_size_mb(model_path),
        **metadata,
    }


def model_onnx_map(layout: OutputLayout) -> dict[str, Path]:
    return {
        "FP32 Research Reference": onnx_path(layout, "fp32_reference.onnx"),
        "Primary Baseline (FP16)": onnx_path(layout, "primary_baseline_fp16.onnx"),
        "INT8 Uniform": onnx_path(layout, "int8_uniform.onnx"),
        "Greedy Mixed": onnx_path(layout, "greedy_mixed.onnx"),
        "SA (v1)": onnx_path(layout, "sa_v1.onnx"),
        "Hybrid SA (ours)": onnx_path(layout, "hybrid_sa.onnx"),
    }


def generate_size_comparison(layout: OutputLayout) -> dict[str, Any]:
    reference_checkpoint = checkpoint_path(layout, "fp32_reference_best.pt")
    if not reference_checkpoint.exists():
        raise FileNotFoundError("Missing FP32 reference checkpoint for size comparison.")
    reference_model = load_model_from_checkpoint(reference_checkpoint, device=torch.device("cpu"), use_half=False)

    greedy_config = load_json(results_path(layout, "greedy_config.json"), default={}) or {}
    sa_payload = load_json(results_path(layout, "sa_best_config.json"), default={}) or {}
    sensitivity_payload = load_json(results_path(layout, "sensitivity_results.json"), default={}) or {}
    hybrid_config = dict(sa_payload.get("default_result", {}).get("bit_config", {}))
    for layer in sensitivity_payload.get("locked_layers", []):
        hybrid_config[layer] = 8

    theoretical_configs = {
        "FP32 Research Reference": {"default_bits": DEFAULT_BITS, "embed_bits": DEFAULT_BITS},
        "Primary Baseline (FP16)": {"default_bits": FP16_BITS, "embed_bits": FP16_BITS},
        "INT8 Uniform": {"default_bits": INT8_BITS, "embed_bits": INT8_BITS},
        "Greedy Mixed": {"default_bits": DEFAULT_BITS, "embed_bits": DEFAULT_BITS, "linear_config": greedy_config},
        "SA (v1)": {"default_bits": DEFAULT_BITS, "embed_bits": DEFAULT_BITS, "linear_config": hybrid_config},
        "Hybrid SA (ours)": {"default_bits": DEFAULT_BITS, "embed_bits": HYBRID_EMBED_BITS, "linear_config": hybrid_config},
    }

    fp32_reference = compute_theoretical_size_mb(reference_model)["theoretical_mb"]
    results: dict[str, Any] = {}
    for method, model_path_item in model_onnx_map(layout).items():
        theoretical = compute_theoretical_size_mb(reference_model, **theoretical_configs[method])
        ratio = fp32_reference / theoretical["theoretical_mb"]
        results[method] = {
            "onnx_file": model_path_item.name,
            "onnx_mb": round(total_onnx_size_mb(model_path_item), 2) if model_path_item.exists() else None,
            "theoretical_mb": round(theoretical["theoretical_mb"], 2),
            "effective_bits": round(theoretical["effective_bits"], 2),
            "total_params": theoretical["total_params"],
            "compression_ratio": round(ratio, 2),
        }

    save_json(results_path(layout, "size_comparison.json"), results)
    lines = [f"{'Method':<28} {'ONNX MB':>10} {'Theory MB':>11} {'Eff Bits':>10} {'Ratio':>8}", "-" * 72]
    for method in METHOD_ORDER:
        payload = results.get(method)
        onnx_value = "N/A" if payload["onnx_mb"] is None else f"{payload['onnx_mb']:.2f}"
        lines.append(
            f"{method:<28} {onnx_value:>10} {payload['theoretical_mb']:>11.2f} "
            f"{payload['effective_bits']:>10.2f} {payload['compression_ratio']:>7.2f}x"
        )
    results_path(layout, "size_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def run_evaluation_phase(layout: OutputLayout, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = build_tokenizer()
    dataset = load_mrpc_dataset(
        tokenizer,
        split=VALIDATION_SPLIT,
        format_type="numpy",
        max_samples=args.max_eval_samples,
    )
    all_results: dict[str, Any] = {}
    for method, model_path_item in model_onnx_map(layout).items():
        print(f"Evaluating {method}: {model_path_item}")
        if not model_path_item.exists():
            all_results[method] = None
            continue
        metrics = evaluate_onnx_model(
            model_path_item,
            dataset,
            batch_size=max(1, min(DEFAULT_EVAL_BATCH_SIZE, args.batch_size)),
            latency_runs=args.latency_runs,
            threads=args.threads,
        )
        all_results[method] = {
            "accuracy": round(metrics["accuracy"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "latency_avg": round(metrics["latency_avg_ms"], 2),
            "latency_p50": round(metrics["latency_p50_ms"], 2),
            "latency_p95": round(metrics["latency_p95_ms"], 2),
            "latency_p99": round(metrics["latency_p99_ms"], 2),
            "onnx_mb": round(metrics["onnx_size_mb"], 2),
            "requested_provider": metrics["requested_provider"],
            "active_providers": metrics["active_providers"],
            "execution_backend": metrics["execution_backend"],
            "fallback_reason": metrics["fallback_reason"],
        }

    save_json(results_path(layout, "all_model_results.json"), all_results)
    lines = [
        f"{'Method':<28} {'Accuracy':>9} {'F1':>7} {'Latency(ms)':>12} {'ONNX(MB)':>10} {'Backend':>18}",
        "-" * 94,
    ]
    for method in METHOD_ORDER:
        metrics = all_results.get(method)
        if metrics is None:
            lines.append(f"{method:<28} {'N/A':>9} {'N/A':>7} {'N/A':>12} {'N/A':>10} {'missing':>18}")
            continue
        lines.append(
            f"{method:<28} {metrics['accuracy']:>9.4f} {metrics['f1']:>7.4f} "
            f"{metrics['latency_avg']:>12.2f} {metrics['onnx_mb']:>10.2f} {metrics['execution_backend']:>18}"
        )
    results_path(layout, "all_model_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    generate_size_comparison(layout)
    return all_results


# EVALUATION_PHASE
def rank_colors(values: list[float], *, higher_is_better: bool) -> list[Any]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    if np.allclose(arr.max(), arr.min()):
        return [cmap(0.5) for _ in values]
    normalized = (arr - arr.min()) / (arr.max() - arr.min())
    if not higher_is_better:
        normalized = 1.0 - normalized
    return [cmap(0.12 + 0.76 * score) for score in normalized]


def annotate_bars(ax: Any, bars: Any, fmt: str) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.015 if y_max > y_min else 0.01
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def generate_training_convergence_figure(layout: OutputLayout) -> None:
    payload = load_json(results_path(layout, "training_history.json"))
    if not payload:
        raise FileNotFoundError("Missing training_history.json")
    runs = payload["runs"]
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    styles = {
        "FP32 Research Reference": {"acc": "#1565C0", "f1": "#2E7D32", "loss": "#EF6C00", "linestyle": "-", "marker": "o"},
        "Primary Baseline (FP16)": {"acc": "#6A1B9A", "f1": "#00897B", "loss": "#C62828", "linestyle": "--", "marker": "s"},
    }
    for label, run in runs.items():
        style = styles.get(label, styles["FP32 Research Reference"])
        history = run["history"]
        steps = [entry["global_step"] for entry in history]
        accs = [entry["validation_accuracy"] for entry in history]
        f1s = [entry["validation_f1"] for entry in history]
        losses = [entry["train_loss"] for entry in history]
        ax_acc.plot(steps, accs, label=f"{label} accuracy", color=style["acc"], linestyle=style["linestyle"], marker=style["marker"])
        ax_acc.plot(steps, f1s, label=f"{label} F1", color=style["f1"], linestyle=style["linestyle"], marker="^")
        ax_loss.plot(steps, losses, label=f"{label} loss", color=style["loss"], linestyle=style["linestyle"], marker=style["marker"])

    ax_acc.set_title("Validation Convergence", fontweight="bold")
    ax_acc.set_xlabel("Cumulative Training Steps")
    ax_acc.set_ylabel("Metric")
    ax_acc.grid(True, alpha=0.25, linestyle="--")
    ax_acc.legend(loc="best")

    ax_loss.set_title("Optimization Trajectory", fontweight="bold")
    ax_loss.set_xlabel("Cumulative Training Steps")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.grid(True, alpha=0.25, linestyle="--")
    ax_loss.legend(loc="best")

    fig.suptitle("ModernBERT-large MRPC Training Convergence", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(figures_path(layout, "training_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_accuracy_vs_size_figure(layout: OutputLayout) -> None:
    all_results = load_json(results_path(layout, "all_model_results.json"))
    size_results = load_json(results_path(layout, "size_comparison.json"))
    if not all_results or not size_results:
        raise FileNotFoundError("Missing all_model_results.json or size_comparison.json")

    points = []
    for method in METHOD_ORDER:
        eval_entry = all_results.get(method)
        size_entry = size_results.get(method)
        if not eval_entry or not size_entry or eval_entry.get("accuracy") is None:
            continue
        points.append((method, size_entry["theoretical_mb"], eval_entry["accuracy"]))

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.scatter(
        [item[1] for item in points],
        [item[2] for item in points],
        s=110,
        c=[METHOD_COLORS.get(item[0], "#455A64") for item in points],
        edgecolors="white",
        linewidths=1.3,
        zorder=3,
    )
    for method, size_mb, accuracy in points:
        ax.annotate(
            method,
            (size_mb, accuracy),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.4,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.74},
        )
    ax.set_xlabel("Theoretical Model Size (MB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("MRPC Accuracy vs Theoretical Model Size", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(figures_path(layout, "accuracy_vs_size_tradeoff.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_compression_bars(layout: OutputLayout) -> None:
    size_results = load_json(results_path(layout, "size_comparison.json"))
    if not size_results:
        raise FileNotFoundError("Missing size_comparison.json")

    fp32_size = size_results["FP32 Research Reference"]["theoretical_mb"]
    methods = [name for name in METHOD_ORDER if size_results.get(name)]
    sizes_theory = [size_results[name]["theoretical_mb"] for name in methods]
    ratios = [fp32_size / size for size in sizes_theory]
    labels = [SHORT_LABELS[name] for name in methods]
    size_colors = rank_colors(sizes_theory, higher_is_better=False)
    ratio_colors = rank_colors(ratios, higher_is_better=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    bars1 = ax1.bar(labels, sizes_theory, color=size_colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax1.axhline(fp32_size, color="black", linestyle="--", linewidth=1.4)
    ax1.set_ylabel("Theoretical Effective Size (MB)")
    ax1.set_title("Model Size", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax1, bars1, "{:.1f} MB")

    bars2 = ax2.bar(labels, ratios, color=ratio_colors, width=0.6, edgecolor="white", linewidth=1.2)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax2.set_ylabel("Compression Ratio (x)")
    ax2.set_title("Compression vs FP32", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax2, bars2, "{:.2f}x")

    fig.suptitle("ModernBERT-large MRPC Compression Overview", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(figures_path(layout, "compression_bars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_latency_bars(layout: OutputLayout) -> None:
    all_results = load_json(results_path(layout, "all_model_results.json"))
    if not all_results:
        raise FileNotFoundError("Missing all_model_results.json")

    methods = [name for name in METHOD_ORDER if all_results.get(name)]
    labels = [SHORT_LABELS[name] for name in methods]
    latencies = [all_results[name]["latency_avg"] for name in methods]
    p95 = [all_results[name]["latency_p95"] for name in methods]
    colors = rank_colors(latencies, higher_is_better=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, latencies, color=colors, width=0.58, edgecolor="white", linewidth=1.1)
    ax.errorbar(
        range(len(methods)),
        latencies,
        yerr=[np.zeros(len(methods)), [max(0.0, hi - lo) for hi, lo in zip(p95, latencies)]],
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.3,
    )
    ax.set_ylabel("Average Inference Latency (ms)")
    ax.set_title("MRPC ONNX Latency Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax, bars, "{:.2f}")
    plt.tight_layout()
    plt.savefig(figures_path(layout, "latency_bars.png"), dpi=200, bbox_inches="tight")
    plt.close()


def generate_accuracy_drop_bars(layout: OutputLayout) -> None:
    all_results = load_json(results_path(layout, "all_model_results.json"))
    if not all_results:
        raise FileNotFoundError("Missing all_model_results.json")

    fp32_accuracy = all_results["FP32 Research Reference"]["accuracy"]
    methods = [name for name in METHOD_ORDER if all_results.get(name) and name != "FP32 Research Reference"]
    labels = [SHORT_LABELS[name] for name in methods]
    drops = [(fp32_accuracy - all_results[name]["accuracy"]) * 100 for name in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, drops, color=[METHOD_COLORS[name] for name in methods], width=0.55, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, color="#FF9800", linestyle="--", linewidth=1.5)
    ax.axhline(2.0, color="#F44336", linestyle="--", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Accuracy Drop vs FP32 (%)")
    ax.set_title("Accuracy Degradation Relative to FP32 Reference", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    annotate_bars(ax, bars, "{:.2f}%")
    plt.tight_layout()
    plt.savefig(figures_path(layout, "accuracy_drop_bars.png"), dpi=200, bbox_inches="tight")
    plt.close()


def parse_block_and_op(layer_name: str) -> tuple[int | None, str]:
    parts = layer_name.split(".")
    block_index = None
    block_pos = -1
    preferred_prefixes = {"layer", "layers", "block", "blocks", "h"}
    for index, part in enumerate(parts):
        if part.isdigit():
            if index > 0 and parts[index - 1] in preferred_prefixes:
                block_index = int(part)
                block_pos = index
                break
            if block_index is None:
                block_index = int(part)
                block_pos = index
    op_suffix = ".".join(parts[block_pos + 1 :]) if block_pos >= 0 else layer_name
    return block_index, op_suffix or layer_name


def pretty_op_label(op_name: str) -> str:
    return op_name.replace("self_attn", "attn").replace("attention", "attn")


def generate_hybrid_heatmap(layout: OutputLayout) -> None:
    summary = load_json(results_path(layout, "hybrid_config_summary.json"))
    if not summary:
        raise FileNotFoundError("Missing hybrid_config_summary.json")

    config = summary.get("hybrid_sa_config") or summary.get("sa_config") or {}
    if not config:
        raise FileNotFoundError("No hybrid SA config found in hybrid_config_summary.json")

    parsed = []
    for layer_name, bits in config.items():
        block_index, op_name = parse_block_and_op(layer_name)
        if block_index is None:
            continue
        parsed.append((block_index, op_name, bits))

    if not parsed:
        raise RuntimeError("Unable to derive block/op structure for the hybrid heatmap.")

    blocks = sorted({block for block, _, _ in parsed})
    ops = sorted({op for _, op, _ in parsed})
    matrix = np.full((len(blocks), len(ops)), np.nan)
    for block_index, op_name, bits in parsed:
        row = blocks.index(block_index)
        col = ops.index(op_name)
        matrix[row, col] = bits

    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(3).copy()
    cmap.set_bad(color="#ECEFF1")
    fig_width = max(10.0, 0.9 * len(ops))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=3, vmax=9, aspect="auto")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = "-" if np.isnan(value) else f"{int(value)}b"
            ax.text(col, row, label, ha="center", va="center", fontsize=10, fontweight="bold", color="black")

    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([pretty_op_label(op) for op in ops], rotation=30, ha="right")
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Block {index}" for index in blocks])
    ax.set_title("Hybrid SA Bit Allocation Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, ticks=[4, 6, 8]).set_label("Bit Width")
    plt.tight_layout()
    plt.savefig(figures_path(layout, "hybrid_sa_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_figures_phase(layout: OutputLayout) -> None:
    generate_training_convergence_figure(layout)
    generate_accuracy_vs_size_figure(layout)
    generate_compression_bars(layout)
    generate_latency_bars(layout)
    generate_accuracy_drop_bars(layout)
    generate_hybrid_heatmap(layout)


def run_all_phases(layout: OutputLayout, args: argparse.Namespace) -> None:
    run_training_phase(layout, args)
    run_search_phase(layout, args)
    run_export_phase(layout, args)
    run_evaluation_phase(layout, args)
    run_figures_phase(layout)


# FIGURES_PHASE


def main() -> None:
    args = parse_args()
    layout = build_layout(args.output_root)

    if args.phase == "all":
        run_all_phases(layout, args)
    elif args.phase == "train":
        run_training_phase(layout, args)
    elif args.phase == "search":
        run_search_phase(layout, args)
    elif args.phase == "export":
        run_export_phase(layout, args)
    elif args.phase == "evaluate":
        run_evaluation_phase(layout, args)
    elif args.phase == "figures":
        run_figures_phase(layout)


if __name__ == "__main__":
    main()
