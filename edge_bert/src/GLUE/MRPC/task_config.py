from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = TASK_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "experiment_settings.json"
MODELS_DIR = TASK_ROOT / "models"
RESULTS_DIR = TASK_ROOT / "results"
FIGURES_DIR = TASK_ROOT / "figures"


DEFAULT_SETTINGS: dict[str, Any] = {
    "model": {
        "name": "distilbert-base-uncased",
        "num_labels": 2,
    },
    "dataset": {
        "name": "glue",
        "config": "mrpc",
        "train_split": "train",
        "validation_split": "validation",
        "text_columns": ["sentence1", "sentence2"],
        "max_length": 128,
        "vocab_size": 30522,
    },
    "training": {
        "active_regimen": "realistic",
        "available_regimens": ["realistic", "controlled"],
        "configs": {
            "realistic": {
                "max_epochs": 20,
                "early_stopping_patience": 2,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "seed": 42,
            },
            "controlled": {
                "max_epochs": 20,
                "early_stopping_patience": None,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "seed": 42,
            },
        },
    },
    "evaluation": {
        "batch_size": 32,
        "latency_runs": 100,
        "threads": 4,
        "max_samples": None,
    },
    "quantization": {
        "bit_options": [4, 6, 8],
        "default_bits": 32,
        "embedding_bits": 8,
    },
    "sensitivity": {
        "bit_simulation": 4,
        "lock_threshold": 0.01,
        "subset_strategy": "fixed_random_cap",
        "subset_size": 2048,
        "subset_seed": 42,
        "results_file": "sensitivity_results.json",
    },
    "search": {
        "alpha": 5.0,
        "beta": 2.0,
        "gamma": 1.0,
        "threshold_drop_percentages": [0.25, 0.5, 0.75, 1.0, 1.25],
        "default_threshold_index": 0,
        "guide_latency_weight": 1.0,
        "guide_size_weight": 1.0,
        "initial_temperature": 1.0,
        "final_temperature": 0.05,
        "cooling_rate": 0.9,
        "iterations_per_temp": 3,
        "early_stop_patience": 8,
        "plateau_eps": 1e-3,
        "num_runs": 3,
        "baseline_size_mb_fallback": 255.55,
        "baseline_latency_ms_fallback": 14.67,
        "results_file": "sa_search_results.json",
        "best_config_file": "sa_best_config.json",
        "search_eval_max_samples": None,
        "search_batch_size": 16,
    },
    "greedy": {
        "allocation_strategy": "tertiles",
    },
    "runtime": {
        "fp16_onnx_provider": "CUDAExecutionProvider",
        "default_onnx_providers": ["CPUExecutionProvider"],
    },
    "tracking": {
        "reference_results_file": "reference_model_results.json",
        "all_model_results_file": "all_model_results.json",
        "all_model_results_txt_file": "all_model_results.txt",
        "fp16_baseline_onnx_metrics_file": "fp16_baseline_onnx_metrics.json",
        "greedy_config_file": "greedy_config.json",
        "hybrid_summary_file": "hybrid_config_summary.json",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    settings = deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, encoding="utf-8") as file:
            settings = _deep_merge(settings, json.load(file))
    return settings


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as file:
        return json.load(file)


MODEL_SETTINGS = get_settings()["model"]
DATASET_SETTINGS = get_settings()["dataset"]
TRAINING_SETTINGS = get_settings()["training"]
EVALUATION_SETTINGS = get_settings()["evaluation"]
QUANTIZATION_SETTINGS = get_settings()["quantization"]
SENSITIVITY_SETTINGS = get_settings()["sensitivity"]
SEARCH_SETTINGS = get_settings()["search"]
GREEDY_SETTINGS = get_settings()["greedy"]
RUNTIME_SETTINGS = get_settings()["runtime"]
TRACKING_SETTINGS = get_settings()["tracking"]

MODEL_NAME = MODEL_SETTINGS["name"]
NUM_LABELS = MODEL_SETTINGS["num_labels"]
MAX_LENGTH = DATASET_SETTINGS["max_length"]
VOCAB_SIZE = DATASET_SETTINGS["vocab_size"]

DATASET_NAME = DATASET_SETTINGS["name"]
DATASET_CONFIG = DATASET_SETTINGS["config"]
TRAIN_SPLIT = DATASET_SETTINGS["train_split"]
VALIDATION_SPLIT = DATASET_SETTINGS["validation_split"]
TEXT_COLUMNS = tuple(DATASET_SETTINGS["text_columns"])

ACTIVE_REGIMEN = TRAINING_SETTINGS["active_regimen"]
AVAILABLE_REGIMENS = tuple(TRAINING_SETTINGS["available_regimens"])
TRAINING_CONFIGS: dict[str, dict[str, Any]] = TRAINING_SETTINGS["configs"]

FP16_ONNX_PROVIDER = RUNTIME_SETTINGS["fp16_onnx_provider"]
FP16_ONNX_PROVIDERS = [FP16_ONNX_PROVIDER]
DEFAULT_ONNX_PROVIDERS = list(RUNTIME_SETTINGS["default_onnx_providers"])

RESULT_FILES: dict[str, str] = {
    "reference_results": TRACKING_SETTINGS["reference_results_file"],
    "all_model_results": TRACKING_SETTINGS["all_model_results_file"],
    "all_model_results_txt": TRACKING_SETTINGS["all_model_results_txt_file"],
    "fp16_baseline_onnx_metrics": TRACKING_SETTINGS["fp16_baseline_onnx_metrics_file"],
    "sensitivity_results": SENSITIVITY_SETTINGS["results_file"],
    "sa_search_results": SEARCH_SETTINGS["results_file"],
    "sa_best_config": SEARCH_SETTINGS["best_config_file"],
    "greedy_config": TRACKING_SETTINGS["greedy_config_file"],
    "hybrid_summary": TRACKING_SETTINGS["hybrid_summary_file"],
}


def get_active_regimen() -> str:
    return ACTIVE_REGIMEN


def training_config(regimen: str | None = None) -> dict[str, Any]:
    target = regimen or get_active_regimen()
    if target not in TRAINING_CONFIGS:
        raise ValueError(f"Unsupported MRPC regimen: {target}")
    return dict(TRAINING_CONFIGS[target])


def baseline_checkpoint_path(regimen: str | None = None) -> Path:
    return MODELS_DIR / (regimen or get_active_regimen()) / "baseline_best.pt"


def baseline_fp32_onnx_path() -> Path:
    return MODELS_DIR / "baseline_fp32.onnx"


def baseline_fp16_onnx_path() -> Path:
    return MODELS_DIR / "baseline_fp16.onnx"


def baseline_int8_onnx_path() -> Path:
    return MODELS_DIR / "baseline_int8.onnx"


def greedy_checkpoint_path() -> Path:
    return MODELS_DIR / "greedy_quant_model.pt"


def greedy_onnx_path() -> Path:
    return MODELS_DIR / "greedy_quant_model.onnx"


def sa_onnx_path() -> Path:
    return MODELS_DIR / "mixed_precision_real_quant.onnx"


def hybrid_onnx_path() -> Path:
    return MODELS_DIR / "hybrid_quant_model.onnx"


def reference_results_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["reference_results"]


def all_model_results_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["all_model_results"]


def all_model_results_txt_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["all_model_results_txt"]


def fp16_baseline_onnx_metrics_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["fp16_baseline_onnx_metrics"]


def sensitivity_results_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["sensitivity_results"]


def sa_search_results_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["sa_search_results"]


def sa_best_config_path() -> Path:
    return RESULTS_DIR / RESULT_FILES["sa_best_config"]


def greedy_config_path() -> Path:
    return MODELS_DIR / RESULT_FILES["greedy_config"]


def hybrid_summary_path() -> Path:
    return MODELS_DIR / RESULT_FILES["hybrid_summary"]


def load_sensitivity_results() -> dict[str, Any] | None:
    return load_json(sensitivity_results_path())


def derive_locked_layers(sensitivity_by_layer: dict[str, float], threshold: float | None = None) -> list[str]:
    lock_threshold = threshold if threshold is not None else SENSITIVITY_SETTINGS["lock_threshold"]
    return sorted(layer for layer, drop in sensitivity_by_layer.items() if drop >= lock_threshold)


def get_locked_layers() -> list[str]:
    sensitivity_results = load_sensitivity_results()
    if not sensitivity_results:
        return []
    if "locked_layers" in sensitivity_results:
        return sorted(sensitivity_results["locked_layers"])
    sensitivity_by_layer = sensitivity_results.get("sensitivity_by_layer", {})
    return derive_locked_layers(sensitivity_by_layer)


def load_greedy_config(required: bool = True) -> dict[str, int]:
    config = load_json(greedy_config_path())
    if isinstance(config, dict):
        return config
    if required:
        raise FileNotFoundError(f"Missing MRPC greedy config: {greedy_config_path()}")
    return {}


def load_sa_best_config(required: bool = True) -> dict[str, int]:
    best_config = load_json(sa_best_config_path())
    if isinstance(best_config, dict):
        if "bit_config" in best_config:
            return best_config["bit_config"]
        if all(isinstance(value, int) for value in best_config.values()):
            return best_config

    hybrid_summary = load_json(hybrid_summary_path())
    if isinstance(hybrid_summary, dict) and "sa_config" in hybrid_summary:
        return hybrid_summary["sa_config"]

    if required:
        raise FileNotFoundError(f"Missing MRPC SA config: {sa_best_config_path()}")
    return {}


def get_search_reference_metrics() -> tuple[float, float]:
    reference_results = load_json(reference_results_path(), default={}) or {}
    fp32_reference = reference_results.get("models", {}).get("FP32 Research Reference", {})
    fp16_reference = reference_results.get("models", {}).get("Primary Baseline (FP16)", {})

    size_mb = (
        fp32_reference.get("model_file_size_mb")
        or fp16_reference.get("model_file_size_mb")
        or (round(baseline_checkpoint_path().stat().st_size / (1024**2), 2) if baseline_checkpoint_path().exists() else None)
        or SEARCH_SETTINGS["baseline_size_mb_fallback"]
    )
    latency_ms = (
        fp32_reference.get("latency_ms")
        or fp16_reference.get("latency_ms")
        or SEARCH_SETTINGS["baseline_latency_ms_fallback"]
    )
    return float(size_mb), float(latency_ms)


def save_sensitivity_results(results: dict[str, Any]) -> None:
    save_json(sensitivity_results_path(), results)


def save_reference_model_results(results: dict[str, Any]) -> None:
    save_json(reference_results_path(), results)


def save_all_model_results(results: dict[str, Any]) -> None:
    save_json(all_model_results_path(), results)


def save_sa_search_results(search_results: dict[str, Any]) -> None:
    save_json(sa_search_results_path(), search_results)

    default_result = search_results.get("default_result") or {}
    candidate = default_result.get("candidate")
    if not candidate:
        return

    save_json(
        sa_best_config_path(),
        {
            "bit_config": candidate,
            "search_mode": search_results.get("search_mode", "constrained_pareto"),
            "baseline_accuracy": search_results.get("baseline_accuracy"),
            "threshold_label": default_result.get("threshold_label"),
            "threshold_accuracy": default_result.get("threshold_accuracy"),
            "threshold_drop_percent": default_result.get("threshold_drop_percent"),
            "best_seed": default_result.get("seed"),
            "guide_energy": default_result.get("guide_energy"),
            "accuracy": default_result.get("accuracy"),
            "size_mb": default_result.get("size"),
            "latency_ms": default_result.get("latency"),
        },
    )


ONNX_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "FP32 Research Reference": {
        "path": baseline_fp32_onnx_path(),
        "providers": DEFAULT_ONNX_PROVIDERS,
    },
    "Primary Baseline (FP16)": {
        "path": baseline_fp16_onnx_path(),
        "providers": FP16_ONNX_PROVIDERS,
        "required_provider": FP16_ONNX_PROVIDER,
    },
    "INT8 Uniform": {
        "path": baseline_int8_onnx_path(),
        "providers": DEFAULT_ONNX_PROVIDERS,
    },
    "Greedy Mixed": {
        "path": greedy_onnx_path(),
        "providers": DEFAULT_ONNX_PROVIDERS,
    },
    "SA (v1)": {
        "path": sa_onnx_path(),
        "providers": DEFAULT_ONNX_PROVIDERS,
    },
    "Hybrid SA (ours)": {
        "path": hybrid_onnx_path(),
        "providers": DEFAULT_ONNX_PROVIDERS,
    },
}
