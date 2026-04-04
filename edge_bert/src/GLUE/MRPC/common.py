from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from GLUE.MRPC.task_config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DEFAULT_ONNX_PROVIDERS,
    EVALUATION_SETTINGS,
    FP16_ONNX_PROVIDER,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_LABELS,
    TEXT_COLUMNS,
    VALIDATION_SPLIT,
    baseline_fp16_onnx_path,
    fp16_baseline_onnx_metrics_path,
    load_json,
    save_json,
)


def ensure_cuda_available(context: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{context} requires CUDA for MRPC FP16 support, but CUDA is not available.")


def ensure_onnx_provider(provider: str) -> None:
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"Required ONNX Runtime provider '{provider}' is not available. "
            f"Available providers: {available}. "
            "Install a GPU-enabled ONNX Runtime build such as `onnxruntime-gpu` that matches your CUDA setup."
        )


def load_mrpc_dataset(
    *,
    split: str = VALIDATION_SPLIT,
    format_type: str = "torch",
    max_samples: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[TEXT_COLUMNS[0]],
            batch[TEXT_COLUMNS[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    label_column = "label"
    if format_type == "torch":
        dataset = dataset.rename_column("label", "labels")
        label_column = "labels"
    dataset.set_format(type=format_type, columns=["input_ids", "attention_mask", label_column])
    return dataset


def load_classifier_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: str = "cpu",
    use_half: bool = False,
) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    if use_half:
        model = model.half()
    model.eval()
    return model


def total_onnx_size_mb(model_path: Path | str) -> float:
    path = Path(model_path)
    size = 0
    if path.exists():
        size += path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        size += sidecar.stat().st_size
    return size / (1024**2)


def cleanup_onnx_artifacts(model_path: Path | str) -> Path:
    path = Path(model_path)
    sidecar = path.with_suffix(path.suffix + ".data")
    if path.exists():
        path.unlink()
    if sidecar.exists():
        sidecar.unlink()
    return path


def consolidate_onnx_to_single_file(model_path: Path | str) -> Path:
    path = Path(model_path)
    sidecar = path.with_suffix(path.suffix + ".data")
    if not sidecar.exists():
        return path

    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path), save_as_external_data=False)
    if sidecar.exists():
        sidecar.unlink()
    return path


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_pytorch_model(
    model: DistilBertForSequenceClassification,
    dataset,
    *,
    latency_runs: int,
) -> dict[str, float]:
    predictions: list[int] = []
    labels: list[int] = []
    probs: list[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for sample in dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.append(torch.argmax(logits, dim=-1).item())
            labels.append(sample["labels"].item())
            probs.append(torch.softmax(logits.float(), dim=-1)[0, 1].item())

    sample = dataset[0]
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
    }
    for _ in range(10):
        with torch.no_grad():
            model(**inputs)

    latencies = []
    for _ in range(latency_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = -1.0

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc,
        "latency_ms": float(np.mean(latencies)),
    }


def build_onnx_session(
    model_path: Path | str,
    *,
    threads: int | None = None,
    providers: list[str] | None = None,
    required_provider: str | None = None,
) -> ort.InferenceSession:
    if required_provider:
        ensure_onnx_provider(required_provider)

    session_options = ort.SessionOptions()
    if threads is not None:
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers or DEFAULT_ONNX_PROVIDERS,
    )


def evaluate_onnx_model(
    model_path: Path | str,
    dataset,
    *,
    batch_size: int,
    latency_runs: int,
    threads: int | None = None,
    providers: list[str] | None = None,
    required_provider: str | None = None,
) -> dict[str, float]:
    session = build_onnx_session(
        model_path,
        threads=threads,
        providers=providers,
        required_provider=required_provider,
    )
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
        batch_probs = _softmax(logits.astype(np.float32))
        predictions.extend(np.argmax(batch_probs, axis=-1).tolist())
        labels.extend(batch["label"].tolist())
        probs.extend(batch_probs[:, 1].tolist())

    sample = dataset[0]
    single_input = {
        "input_ids": sample["input_ids"][np.newaxis].astype(np.int64),
        "attention_mask": sample["attention_mask"][np.newaxis].astype(np.int64),
    }
    for _ in range(10):
        session.run(["logits"], single_input)

    latencies = []
    for _ in range(latency_runs):
        start = time.perf_counter()
        session.run(["logits"], single_input)
        latencies.append((time.perf_counter() - start) * 1000)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = -1.0

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc,
        "onnx_size_mb": total_onnx_size_mb(model_path),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


class ExportWrapper(nn.Module):
    """
    Export-friendly wrapper.

    Transformers 5.5 builds a bidirectional attention mask internally via `sdpa_mask`, which can
    crash under TorchScript-based ONNX tracing. For the FP32 export path we mirror the SST-2
    exporter and always provide a precomputed float32 4D mask.
    """

    def __init__(self, model: DistilBertForSequenceClassification, *, mask_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = model
        self.mask_dtype = mask_dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_bool = attention_mask > 0
        pairwise = attn_bool[:, None, :, None] & attn_bool[:, None, None, :]
        min_val = torch.finfo(self.mask_dtype).min
        mask = torch.where(pairwise, torch.zeros_like(pairwise, dtype=self.mask_dtype), min_val)
        return self.model(input_ids=input_ids, attention_mask=mask).logits


def export_checkpoint_to_onnx(
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    use_half: bool = False,
) -> Path:
    export_device = "cpu"
    mask_dtype = torch.float32
    if use_half:
        ensure_cuda_available("MRPC FP16 ONNX export")
        export_device = "cuda"
        mask_dtype = torch.float16

    model = load_classifier_checkpoint(
        checkpoint_path,
        device=export_device,
        use_half=use_half,
    )
    wrapped = ExportWrapper(model, mask_dtype=mask_dtype).to(export_device).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy = tokenizer(
        "Question A?",
        "Question B?",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    dummy_ids = dummy["input_ids"].to(export_device)
    dummy_mask = dummy["attention_mask"].to(export_device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_onnx_artifacts(output_path)
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
    return output_path


def compute_fp16_baseline_onnx_metrics(*, overwrite: bool = False) -> dict[str, Any]:
    metrics_path = fp16_baseline_onnx_metrics_path()
    if metrics_path.exists() and not overwrite:
        cached = load_json(metrics_path, default={}) or {}
        if cached:
            return cached

    metrics = evaluate_onnx_model(
        baseline_fp16_onnx_path(),
        load_mrpc_dataset(format_type="numpy", max_samples=EVALUATION_SETTINGS["max_samples"]),
        batch_size=EVALUATION_SETTINGS["batch_size"],
        latency_runs=EVALUATION_SETTINGS["latency_runs"],
        threads=EVALUATION_SETTINGS["threads"],
        providers=[FP16_ONNX_PROVIDER],
        required_provider=FP16_ONNX_PROVIDER,
    )
    payload = {
        "model": str(baseline_fp16_onnx_path()),
        "accuracy": round(metrics["accuracy"], 4),
        "f1": round(metrics["f1"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "onnx_size_mb": round(metrics["onnx_size_mb"], 2),
        "latency_avg_ms": round(metrics["latency_avg_ms"], 2),
        "latency_p50_ms": round(metrics["latency_p50_ms"], 2),
        "latency_p95_ms": round(metrics["latency_p95_ms"], 2),
        "latency_p99_ms": round(metrics["latency_p99_ms"], 2),
        "max_samples": EVALUATION_SETTINGS["max_samples"],
        "batch_size": EVALUATION_SETTINGS["batch_size"],
        "threads": EVALUATION_SETTINGS["threads"],
    }
    save_json(metrics_path, payload)
    return payload

