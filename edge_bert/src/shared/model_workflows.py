from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from shared.experiment_settings import (
    DATASET_SETTINGS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
    SRC_ROOT,
)


def resolve_in_src(*parts: str) -> Path:
    return SRC_ROOT.joinpath(*parts)


def load_sst2_validation_dataset(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_samples: int | None = None,
    format_type: str = "torch",
):
    dataset = load_dataset(
        DATASET_SETTINGS["name"],
        DATASET_SETTINGS["config"],
        split=DATASET_SETTINGS["evaluation_split"],
    )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["sentence"],
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
    checkpoint_path: os.PathLike[str] | str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=DEFAULT_NUM_LABELS)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_pytorch_model(
    model: DistilBertForSequenceClassification,
    dataset,
    *,
    latency_runs: int = 100,
) -> dict[str, float]:
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for sample in dataset:
            input_ids = sample["input_ids"].unsqueeze(0)
            attention_mask = sample["attention_mask"].unsqueeze(0)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(torch.argmax(output.logits, dim=-1).item())
            labels.append(sample["labels"].item())

    sample = dataset[0]
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0),
        "attention_mask": sample["attention_mask"].unsqueeze(0),
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

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "latency_ms": float(np.mean(latencies)),
    }


def build_onnx_session(
    model_path: os.PathLike[str] | str,
    *,
    threads: int | None = None,
) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    if threads is not None:
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_onnx_model(
    model_path: os.PathLike[str] | str,
    dataset,
    *,
    batch_size: int = 1,
    latency_runs: int = 100,
    warmup_runs: int = 5,
    threads: int | None = None,
) -> dict[str, float]:
    session = build_onnx_session(model_path, threads=threads)
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
        batch_probs = softmax(logits)
        predictions.extend(np.argmax(batch_probs, axis=-1).tolist())
        labels.extend(batch["label"].tolist())
        probs.extend(batch_probs[:, 1].tolist())

    sample = dataset[0]
    single_input = {
        "input_ids": sample["input_ids"][np.newaxis].astype(np.int64),
        "attention_mask": sample["attention_mask"][np.newaxis].astype(np.int64),
    }
    for _ in range(warmup_runs):
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
        "onnx_size_mb": Path(model_path).stat().st_size / (1024**2),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def export_checkpoint_to_onnx(
    checkpoint_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Path:
    model = load_classifier_checkpoint(checkpoint_path, model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dummy = tokenizer(
        "This is a sample sentence for ONNX export.",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    return output_path


def quantize_onnx_model(
    fp32_model_path: os.PathLike[str] | str,
    int8_model_path: os.PathLike[str] | str,
) -> dict[str, float]:
    int8_model_path = Path(int8_model_path)
    int8_model_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_model_path),
        model_output=str(int8_model_path),
        weight_type=QuantType.QInt8,
    )
    fp32_size = Path(fp32_model_path).stat().st_size / (1024**2)
    int8_size = int8_model_path.stat().st_size / (1024**2)
    return {
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "size_reduction_pct": (1 - int8_size / fp32_size) * 100,
    }
