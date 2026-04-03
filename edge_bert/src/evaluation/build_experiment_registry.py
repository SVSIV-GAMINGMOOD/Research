from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    MODELS_DIR,
    RESULTS_DIR,
    SRC_ROOT,
    TRACKING_SETTINGS,
    baseline_checkpoint_path,
    frozen_checkpoint_path,
    load_all_model_results,
    load_reference_model_results,
    load_size_comparison_results,
    save_experiment_registry,
    save_gguf_runtime_results,
)


ONNX_MODEL_PATHS = {
    "FP32 Baseline": MODELS_DIR / "baseline_fp32.onnx",
    "INT8 Uniform": MODELS_DIR / "baseline_int8.onnx",
    "FAR Frozen FP32": MODELS_DIR / "frozen_fp32.onnx",
    "FAR Frozen INT8": MODELS_DIR / "frozen_int8.onnx",
    "Greedy Mixed": MODELS_DIR / "greedy_quant_model.onnx",
    "SA Mixed (v1)": MODELS_DIR / "mixed_precision_real_quant.onnx",
    "Hybrid INT8+SA (Ours)": MODELS_DIR / "hybrid_quant_model.onnx",
}

CHECKPOINT_PATHS = {
    "Baseline Checkpoint": baseline_checkpoint_path(),
    "Frozen Checkpoint": frozen_checkpoint_path(),
    "Greedy Quantized Checkpoint": MODELS_DIR / "greedy_quant_model.pt",
    "Legacy Baseline INT8 Checkpoint": MODELS_DIR / "baseline_int8.pt",
}

CONFIG_PATHS = {
    "Greedy Config": MODELS_DIR / "greedy_config.json",
    "Hybrid Config Summary": MODELS_DIR / "hybrid_config_summary.json",
    "HF Export Directory": MODELS_DIR / "distilbert_hf_format",
}


def artifact_info(path: Path) -> dict[str, object]:
    exists = path.exists()
    info: dict[str, object] = {
        "path": str(path.relative_to(SRC_ROOT)),
        "exists": exists,
        "artifact_type": "directory" if exists and path.is_dir() else "file",
    }
    if exists and path.is_file():
        info["size_mb"] = round(path.stat().st_size / (1024**2), 2)
    return info


def results_inventory() -> list[dict[str, object]]:
    inventory = []
    for path in sorted(RESULTS_DIR.iterdir(), key=lambda item: item.name.lower()):
        info = artifact_info(path)
        info["name"] = path.name
        inventory.append(info)
    return inventory


def build_gguf_runtime_placeholder() -> dict[str, object]:
    models = {
        "Greedy Mixed": {
            "status": "pending_conversion_and_benchmark",
            "role": "supplementary_runtime_only",
            "source_artifacts": {
                "checkpoint": str((MODELS_DIR / "greedy_quant_model.pt").relative_to(MODELS_DIR.parents[1])),
                "onnx": str((MODELS_DIR / "greedy_quant_model.onnx").relative_to(MODELS_DIR.parents[1])),
                "tensor_map_hint": "src/artifacts/tensor_types_greedy.txt",
            },
            "runtime_metrics": None,
        },
        "SA Mixed (v1)": {
            "status": "pending_conversion_and_benchmark",
            "role": "supplementary_runtime_only",
            "source_artifacts": {
                "onnx": str((MODELS_DIR / "mixed_precision_real_quant.onnx").relative_to(MODELS_DIR.parents[1])),
                "sa_config": "src/results/sa_best_config.json",
                "tensor_map_hint": "custom tensor map still needed for standalone SA GGUF export",
            },
            "runtime_metrics": None,
        },
        "Hybrid INT8+SA (Ours)": {
            "status": "pending_conversion_and_benchmark",
            "role": "supplementary_runtime_only",
            "source_artifacts": {
                "onnx": str((MODELS_DIR / "hybrid_quant_model.onnx").relative_to(MODELS_DIR.parents[1])),
                "summary": str((MODELS_DIR / "hybrid_config_summary.json").relative_to(MODELS_DIR.parents[1])),
                "tensor_map_hint": "src/artifacts/tensor_types_hybrid.txt",
            },
            "runtime_metrics": None,
        },
    }
    return {
        "backend": TRACKING_SETTINGS["supplementary_runtime_backend"],
        "notes": [
            "Use llama.cpp / GGUF as supplementary runtime benchmarking only.",
            "Do not replace ONNX-based classification metrics with llama.cpp throughput results.",
            "Benchmark the same mixed-precision policies only after conversion fidelity is validated.",
        ],
        "models": models,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reference_results = load_reference_model_results()
    onnx_results = load_all_model_results()
    size_results = load_size_comparison_results()
    gguf_runtime_results = build_gguf_runtime_placeholder()

    save_gguf_runtime_results(gguf_runtime_results)

    registry = {
        "strategy": {
            "primary_metrics_backend": TRACKING_SETTINGS["primary_metrics_backend"],
            "supplementary_runtime_backend": TRACKING_SETTINGS["supplementary_runtime_backend"],
            "primary_baseline_precision": TRACKING_SETTINGS["primary_baseline_precision"],
            "reference_precision": TRACKING_SETTINGS["reference_precision"],
            "optional_fp8_models": TRACKING_SETTINGS["optional_fp8_models"],
        },
        "result_sources": {
            "reference_results": "src/results/reference_model_results.json",
            "onnx_results": "src/results/all_model_results.json",
            "size_results": "src/results/size_comparison.json",
            "gguf_runtime_results": "src/results/gguf_runtime_results.json",
            "hybrid_specific_results": "src/results/hybrid_eval_results.json",
        },
        "results_inventory": results_inventory(),
        "reference_models": reference_results.get("models", {}),
        "onnx_models": {},
        "checkpoint_artifacts": {name: artifact_info(path) for name, path in CHECKPOINT_PATHS.items()},
        "config_artifacts": {name: artifact_info(path) for name, path in CONFIG_PATHS.items()},
        "supplementary_runtime": gguf_runtime_results,
        "optional_fp8_track": {
            "status": "planned_optional_experiment",
            "notes": "FP8 experiments are reserved for Greedy Mixed, SA Mixed (v1), and Hybrid INT8+SA (Ours). They are not part of the current mandatory result set.",
            "models": {
                model_name: {"status": "not_implemented_in_repo_yet"}
                for model_name in TRACKING_SETTINGS["optional_fp8_models"]
            },
        },
    }

    for model_name, model_path in ONNX_MODEL_PATHS.items():
        registry["onnx_models"][model_name] = {
            "artifact": artifact_info(model_path),
            "metrics": onnx_results.get(model_name),
            "size_summary": size_results.get(model_name),
        }

    save_experiment_registry(registry)

    print("Saved experiment registry and GGUF runtime placeholder results.")
    print("Primary metrics backend:", TRACKING_SETTINGS["primary_metrics_backend"])
    print("Supplementary runtime backend:", TRACKING_SETTINGS["supplementary_runtime_backend"])


if __name__ == "__main__":
    main()
