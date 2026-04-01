from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SRC_ROOT / "config"
MODELS_DIR = SRC_ROOT / "models"
RESULTS_DIR = SRC_ROOT / "results"
FIGURES_DIR = SRC_ROOT / "figures"
SETTINGS_PATH = CONFIG_DIR / "experiment_settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "model": {"name": "distilbert-base-uncased", "num_labels": 2},
    "dataset": {
        "name": "glue",
        "config": "sst2",
        "evaluation_split": "validation",
        "max_length": 128,
        "default_max_samples": None,
    },
    "quantization": {"bit_options": [4, 6, 8], "default_bits": 32, "embedding_bits": 8},
    "sensitivity": {
        "bit_simulation": 4,
        "lock_threshold": 0.01,
        "fallback_locked_layers": [],
        "results_file": "sensitivity_results.json",
    },
    "search": {
        "alpha": 5.0,
        "beta": 2.0,
        "gamma": 1.0,
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
    },
    "greedy": {"allocation_strategy": "tertiles"},
    "evaluation": {"batch_size": 32, "latency_runs": 100, "threads": 4, "max_samples": None},
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
QUANTIZATION_SETTINGS = get_settings()["quantization"]
SENSITIVITY_SETTINGS = get_settings()["sensitivity"]
SEARCH_SETTINGS = get_settings()["search"]
EVALUATION_SETTINGS = get_settings()["evaluation"]
GREEDY_SETTINGS = get_settings()["greedy"]

DEFAULT_MODEL_NAME = MODEL_SETTINGS["name"]
DEFAULT_NUM_LABELS = MODEL_SETTINGS["num_labels"]
DEFAULT_MAX_LENGTH = DATASET_SETTINGS["max_length"]
DEFAULT_BIT_OPTIONS = QUANTIZATION_SETTINGS["bit_options"]
DEFAULT_EMBEDDING_BITS = QUANTIZATION_SETTINGS["embedding_bits"]
DEFAULT_BITS = QUANTIZATION_SETTINGS["default_bits"]


def sensitivity_results_path() -> Path:
    return RESULTS_DIR / SENSITIVITY_SETTINGS["results_file"]


def sa_search_results_path() -> Path:
    return RESULTS_DIR / SEARCH_SETTINGS["results_file"]


def sa_best_config_path() -> Path:
    return RESULTS_DIR / SEARCH_SETTINGS["best_config_file"]


def load_sensitivity_results() -> dict[str, Any] | None:
    return load_json(sensitivity_results_path())


def derive_locked_layers(sensitivity_by_layer: dict[str, float], threshold: float | None = None) -> list[str]:
    lock_threshold = threshold if threshold is not None else SENSITIVITY_SETTINGS["lock_threshold"]
    return sorted(layer for layer, drop in sensitivity_by_layer.items() if drop >= lock_threshold)


def get_locked_layers() -> list[str]:
    sensitivity_results = load_sensitivity_results()
    if sensitivity_results:
        if "locked_layers" in sensitivity_results:
            return sorted(sensitivity_results["locked_layers"])
        sensitivity_by_layer = sensitivity_results.get("sensitivity_by_layer", {})
        if sensitivity_by_layer:
            return derive_locked_layers(sensitivity_by_layer)
    return sorted(SENSITIVITY_SETTINGS.get("fallback_locked_layers", []))


def load_sa_best_config(required: bool = True) -> dict[str, int]:
    best_config = load_json(sa_best_config_path())
    if isinstance(best_config, dict):
        if "bit_config" in best_config:
            return best_config["bit_config"]
        if "candidate" in best_config:
            return best_config["candidate"]
        if all(isinstance(value, int) for value in best_config.values()):
            return best_config

    hybrid_summary = load_json(MODELS_DIR / "hybrid_config_summary.json")
    if isinstance(hybrid_summary, dict) and "sa_config" in hybrid_summary:
        return hybrid_summary["sa_config"]

    if required:
        raise FileNotFoundError(
            f"Missing SA config. Expected {sa_best_config_path()} or {MODELS_DIR / 'hybrid_config_summary.json'}."
        )
    return {}


def load_greedy_config(required: bool = True) -> dict[str, int]:
    config = load_json(MODELS_DIR / "greedy_config.json")
    if isinstance(config, dict):
        return config
    if required:
        raise FileNotFoundError(f"Missing greedy config: {MODELS_DIR / 'greedy_config.json'}")
    return {}


def load_all_model_results() -> dict[str, Any]:
    return load_json(RESULTS_DIR / "all_model_results.json", default={}) or {}


def load_size_comparison_results() -> dict[str, Any]:
    return load_json(RESULTS_DIR / "size_comparison.json", default={}) or {}


def get_energy_weights() -> tuple[float, float, float]:
    return SEARCH_SETTINGS["alpha"], SEARCH_SETTINGS["beta"], SEARCH_SETTINGS["gamma"]


def get_baseline_reference_metrics() -> tuple[float, float]:
    size_results = load_size_comparison_results()
    fp32_size_mb = (
        size_results.get("FP32 Baseline", {}).get("theoretical_mb")
        or SEARCH_SETTINGS["baseline_size_mb_fallback"]
    )

    eval_results = load_all_model_results()
    int8_latency_ms = (
        eval_results.get("INT8 Uniform", {}).get("latency_avg")
        or SEARCH_SETTINGS["baseline_latency_ms_fallback"]
    )
    return fp32_size_mb, int8_latency_ms


def save_sensitivity_results(results: dict[str, Any]) -> None:
    save_json(sensitivity_results_path(), results)


def save_sa_search_results(best_result: dict[str, Any], all_results: list[dict[str, Any]]) -> None:
    payload = {
        "settings": {
            "search": SEARCH_SETTINGS,
            "bit_options": DEFAULT_BIT_OPTIONS,
            "locked_layers": get_locked_layers(),
        },
        "best_result": best_result,
        "runs": all_results,
    }
    save_json(sa_search_results_path(), payload)
    save_json(
        sa_best_config_path(),
        {
            "bit_config": best_result["candidate"],
            "best_seed": best_result["seed"],
            "energy": best_result["energy"],
            "accuracy": best_result["accuracy"],
            "size_mb": best_result["size"],
            "latency_ms": best_result["latency"],
        },
    )
