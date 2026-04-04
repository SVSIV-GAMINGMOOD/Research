from __future__ import annotations

import json
from pathlib import Path
import re
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
    "SA (v1)": MODELS_DIR / "mixed_precision_real_quant.onnx",
    "Hybrid SA (ours)": MODELS_DIR / "hybrid_quant_model.onnx",
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

PROJECT_ROOT = SRC_ROOT.parent
GGUF_ARTIFACTS_DIR = SRC_ROOT / "llama.cpp"
GGUF_BENCHMARKS_DIR = GGUF_ARTIFACTS_DIR / "benchmarks"
BENCHMARK_DIR_PATTERN = re.compile(r"(?P<stamp>\d{8}_\d{6})_(?P<model>.+)")
BUILD_PATTERN = re.compile(r"^build:\s*(.+)$", re.MULTILINE)
FILE_SIZE_PATTERN = re.compile(r"print_info:\s+file size\s+=\s+([0-9.]+)\s+MiB")
TENSOR_TYPE_PATTERN = re.compile(r"llama_model_loader:\s+- type\s+([A-Za-z0-9_]+):\s+(\d+)\s+tensors")

GGUF_MODEL_SPECS = {
    "distilbert-f16": {
        "display_name": "Primary Baseline (FP16 GGUF)",
        "role": "supplementary_runtime_baseline",
        "gguf_artifact": GGUF_ARTIFACTS_DIR / "distilbert-f16.gguf",
        "source_artifacts": {
            "hf_export_dir": "src/models/distilbert_hf_format",
            "gguf": "src/llama.cpp/distilbert-f16.gguf",
        },
    },
    "distilbert-greedy": {
        "display_name": "Greedy Mixed",
        "role": "supplementary_runtime_only",
        "gguf_artifact": GGUF_ARTIFACTS_DIR / "distilbert-greedy.gguf",
        "source_artifacts": {
            "checkpoint": "src/models/greedy_quant_model.pt",
            "onnx": "src/models/greedy_quant_model.onnx",
            "tensor_map_hint": "src/artifacts/tensor_types_greedy.txt",
            "gguf": "src/llama.cpp/distilbert-greedy.gguf",
        },
    },
    "distilbert-sa": {
        "display_name": "SA (v1)",
        "role": "supplementary_runtime_only",
        "gguf_artifact": GGUF_ARTIFACTS_DIR / "distilbert-sa.gguf",
        "source_artifacts": {
            "onnx": "src/models/mixed_precision_real_quant.onnx",
            "sa_config": "src/results/sa_best_config.json",
            "tensor_map_hint": "custom tensor map still needed for standalone SA GGUF export",
            "gguf": "src/llama.cpp/distilbert-sa.gguf",
        },
    },
    "distilbert-hybrid": {
        "display_name": "Hybrid SA (ours)",
        "role": "supplementary_runtime_only",
        "gguf_artifact": GGUF_ARTIFACTS_DIR / "distilbert-hybrid.gguf",
        "source_artifacts": {
            "onnx": "src/models/hybrid_quant_model.onnx",
            "summary": "src/models/hybrid_config_summary.json",
            "tensor_map_hint": "src/artifacts/tensor_types_hybrid.txt",
            "gguf": "src/llama.cpp/distilbert-hybrid.gguf",
        },
    },
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


def relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def summarize_runtime_measurements(entries: list[dict[str, object]]) -> dict[str, object] | None:
    if not entries:
        return None

    measurements = []
    for entry in sorted(entries, key=lambda item: (int(item["context"]), int(item["threads"]))):
        measurements.append(
            {
                "threads": int(entry["threads"]),
                "context": int(entry["context"]),
                "tokens_per_sec": round(float(entry["tokens_per_sec"]), 2),
                "prompt_eval_time_ms": round(float(entry["prompt_eval_time_ms"]), 2),
                "total_time_ms": round(float(entry["total_time_ms"]), 2),
                "status": entry["status"],
            }
        )

    best_tokens = max(measurements, key=lambda item: item["tokens_per_sec"])
    fastest_total = min(measurements, key=lambda item: item["total_time_ms"])
    average_tokens = round(sum(item["tokens_per_sec"] for item in measurements) / len(measurements), 2)
    average_total_time = round(sum(item["total_time_ms"] for item in measurements) / len(measurements), 2)

    return {
        "measurement_count": len(measurements),
        "average_tokens_per_sec": average_tokens,
        "average_total_time_ms": average_total_time,
        "best_tokens_per_sec": best_tokens["tokens_per_sec"],
        "best_tokens_setting": {
            "threads": best_tokens["threads"],
            "context": best_tokens["context"],
        },
        "fastest_total_time_ms": fastest_total["total_time_ms"],
        "fastest_total_time_setting": {
            "threads": fastest_total["threads"],
            "context": fastest_total["context"],
        },
        "measurements": measurements,
    }


def load_benchmark_run(run_dir: Path) -> dict[str, object] | None:
    match = BENCHMARK_DIR_PATTERN.fullmatch(run_dir.name)
    if match is None:
        return None

    results_path = run_dir / "results.json"
    log_path = run_dir / "last_command_output.log"
    if not results_path.exists() or not log_path.exists():
        return None

    entries = json.loads(results_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    build_match = BUILD_PATTERN.search(log_text)
    file_size_match = FILE_SIZE_PATTERN.search(log_text)
    tensor_types = {
        tensor_type: int(count)
        for tensor_type, count in TENSOR_TYPE_PATTERN.findall(log_text)
    }

    cuda_verified = "ggml_cuda_init: found" in log_text and "using device CUDA0" in log_text
    gpu_warning = "no usable GPU found" in log_text or "compiled without GPU support" in log_text

    return {
        "model_key": match.group("model"),
        "run_stamp": match.group("stamp"),
        "benchmark_dir": relative_to_project(run_dir),
        "results_file": relative_to_project(results_path),
        "log_file": relative_to_project(log_path),
        "build": build_match.group(1) if build_match else None,
        "cuda_verified": cuda_verified,
        "gpu_warning": gpu_warning,
        "gguf_file_size_mb": round(float(file_size_match.group(1)), 2) if file_size_match else None,
        "tensor_types": tensor_types,
        "cpu": summarize_runtime_measurements([entry for entry in entries if entry["mode"] == "cpu"]),
        "cuda": summarize_runtime_measurements([entry for entry in entries if entry["mode"] == "gpu"]),
    }


def build_run_summary(run: dict[str, object]) -> dict[str, object]:
    summary = {
        "run_stamp": run["run_stamp"],
        "benchmark_dir": run["benchmark_dir"],
        "results_file": run["results_file"],
        "log_file": run["log_file"],
        "build": run["build"],
        "gguf_file_size_mb": run["gguf_file_size_mb"],
        "tensor_types": run["tensor_types"],
        "devices": {
            "cpu": run["cpu"],
        },
    }

    if run["cuda_verified"]:
        summary["devices"]["cuda"] = run["cuda"]
        summary["validation"] = {
            "cpu_verified": True,
            "cuda_verified": True,
        }
    else:
        summary["validation"] = {
            "cpu_verified": True,
            "cuda_verified": False,
            "note": "GPU-flagged rows are excluded from canonical CUDA reporting because this run was built without usable GPU support.",
        }

    return summary


def build_archived_run_summary(run: dict[str, object]) -> dict[str, object]:
    summary = {
        "run_stamp": run["run_stamp"],
        "benchmark_dir": run["benchmark_dir"],
        "results_file": run["results_file"],
        "log_file": run["log_file"],
        "build": run["build"],
    }
    if run["gpu_warning"]:
        summary["status"] = "archived_cpu_only_build"
        summary["note"] = "The benchmark requested GPU mode, but llama.cpp reported no usable GPU support and ignored GPU offload."
    else:
        summary["status"] = "archived_superseded_run"
    return summary


def build_gguf_runtime_results() -> dict[str, object]:
    notes = [
        "Use llama.cpp / GGUF as supplementary runtime benchmarking only.",
        "Do not replace ONNX-based classification metrics with llama.cpp throughput results.",
        "Benchmark the same mixed-precision policies only after conversion fidelity is validated.",
    ]

    discovered_runs: dict[str, list[dict[str, object]]] = {key: [] for key in GGUF_MODEL_SPECS}
    if GGUF_BENCHMARKS_DIR.exists():
        for run_dir in sorted(path for path in GGUF_BENCHMARKS_DIR.iterdir() if path.is_dir()):
            run = load_benchmark_run(run_dir)
            if run is None or run["model_key"] not in discovered_runs:
                continue
            discovered_runs[run["model_key"]].append(run)

    invalid_gpu_runs = [
        run
        for runs in discovered_runs.values()
        for run in runs
        if run["gpu_warning"] and not run["cuda_verified"]
    ]
    if invalid_gpu_runs:
        invalid_run_stamps = ", ".join(sorted({str(run["run_stamp"]) for run in invalid_gpu_runs}))
        notes.append(
            f"Archived benchmark batch(es) {invalid_run_stamps} are preserved for provenance, but their GPU-labelled rows are not treated as CUDA results because llama.cpp was compiled without usable GPU support."
        )

    models: dict[str, dict[str, object]] = {}
    for model_key, spec in GGUF_MODEL_SPECS.items():
        runs = sorted(discovered_runs[model_key], key=lambda item: item["run_stamp"])
        canonical_run = next((run for run in reversed(runs) if run["cuda_verified"]), None)
        if canonical_run is None and runs:
            canonical_run = runs[-1]

        model_entry = {
            "status": "pending_conversion_and_benchmark",
            "role": spec["role"],
            "source_artifacts": spec["source_artifacts"],
            "gguf_artifact": artifact_info(spec["gguf_artifact"]),
            "runtime_metrics": None,
        }

        if canonical_run is not None:
            model_entry["status"] = (
                "benchmarked_cpu_and_cuda"
                if canonical_run["cuda_verified"]
                else "benchmarked_cpu_only"
            )
            model_entry["runtime_metrics"] = build_run_summary(canonical_run)

        archived_runs = [build_archived_run_summary(run) for run in runs if run is not canonical_run]
        if archived_runs:
            model_entry["archived_runs"] = archived_runs

        models[spec["display_name"]] = model_entry

    return {
        "backend": TRACKING_SETTINGS["supplementary_runtime_backend"],
        "notes": notes,
        "models": models,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reference_results = load_reference_model_results()
    onnx_results = load_all_model_results()
    size_results = load_size_comparison_results()
    gguf_runtime_results = build_gguf_runtime_results()

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
            "notes": "FP8 experiments are reserved for Greedy Mixed, SA (v1), and Hybrid SA (ours). They are not part of the current mandatory result set.",
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

    print("Saved experiment registry and GGUF runtime benchmark results.")
    print("Primary metrics backend:", TRACKING_SETTINGS["primary_metrics_backend"])
    print("Supplementary runtime backend:", TRACKING_SETTINGS["supplementary_runtime_backend"])


if __name__ == "__main__":
    main()

