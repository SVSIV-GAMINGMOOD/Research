from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    RESULTS_DIR,
    TRACKING_SETTINGS,
    baseline_checkpoint_path,
    frozen_checkpoint_path,
    save_reference_model_results,
)


def build_unavailable_entry(reason: str) -> dict[str, object]:
    return {
        "status": "not_measured",
        "reason": reason,
        "backend": "pytorch",
        "accuracy": None,
        "f1": None,
        "latency_ms": None,
        "model_file_size_mb": None,
    }


def relative_checkpoint_label(path: Path) -> str:
    return str(path.relative_to(path.parents[2])).replace("\\", "/")


def build_pending_entry(checkpoint_name: str, model_size_mb: float | None, *, dtype_label: str) -> dict[str, object]:
    return {
        "status": "pending_manual_run",
        "reason": "Evaluation intentionally deferred. Run this script manually when you want checkpoint-level reference metrics.",
        "backend": "pytorch",
        "dtype": dtype_label,
        "accuracy": None,
        "f1": None,
        "latency_ms": None,
        "model_file_size_mb": model_size_mb,
        "checkpoint": checkpoint_name,
    }


def evaluate_checkpoint(checkpoint_path: Path, *, dtype_label: str = "fp32", use_half: bool = False) -> dict[str, object]:
    from shared.model_workflows import (
        evaluate_pytorch_model,
        load_classifier_checkpoint,
        load_sst2_validation_dataset,
    )

    if not checkpoint_path.exists():
        return build_unavailable_entry(f"Missing checkpoint: {checkpoint_path}")

    if use_half and not torch.cuda.is_available():
        return build_unavailable_entry("FP16 checkpoint evaluation requires a CUDA runtime in this project.")

    dataset = load_sst2_validation_dataset(format_type="torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_classifier_checkpoint(checkpoint_path, device=device)
    model.eval()
    if use_half:
        model = model.half()

    with torch.no_grad():
        metrics = evaluate_pytorch_model(model, dataset)
    return {
        "status": "measured",
        "backend": "pytorch",
        "dtype": dtype_label,
        "checkpoint": relative_checkpoint_label(checkpoint_path),
        "device": device,
        "accuracy": round(metrics["accuracy"], 4),
        "f1": round(metrics["f1"], 4),
        "latency_ms": round(metrics["latency_ms"], 2),
        "model_file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 2),
    }


def build_pending_results() -> dict[str, object]:
    checkpoint_map = {
        "Primary Baseline (FP16)": (baseline_checkpoint_path(), "fp16"),
        "FP32 Research Reference": (baseline_checkpoint_path(), "fp32"),
        "Frozen Checkpoint (FP32)": (frozen_checkpoint_path(), "fp32"),
    }
    pending_results = {}
    for model_name, (checkpoint_path, dtype_label) in checkpoint_map.items():
        size_mb = round(checkpoint_path.stat().st_size / (1024**2), 2) if checkpoint_path.exists() else None
        pending_results[model_name] = build_pending_entry(
            relative_checkpoint_label(checkpoint_path),
            size_mb,
            dtype_label=dtype_label,
        )
    return pending_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate or stage reference checkpoint results.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write pending-manual-run placeholders without executing model evaluation.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import_error_reason = None
    try:
        from shared.experiment_settings import MODELS_DIR
    except ImportError as exc:
        MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
        import_error_reason = f"Environment import error prevented PyTorch reference evaluation: {exc}"

    results = {
        "strategy": {
            "primary_baseline_precision": TRACKING_SETTINGS["primary_baseline_precision"],
            "reference_precision": TRACKING_SETTINGS["reference_precision"],
            "notes": "FP16 is the intended modern deployment baseline. FP32 is kept as a research reference point.",
        },
        "models": {},
    }

    if args.prepare_only:
        results["models"] = build_pending_results()
    elif import_error_reason is not None:
        results["models"] = {
            "Primary Baseline (FP16)": build_unavailable_entry(
                "FP16 checkpoint evaluation requires a supported runtime and a working transformers environment."
            ),
            "FP32 Research Reference": build_unavailable_entry(import_error_reason),
            "Frozen Checkpoint (FP32)": build_unavailable_entry(import_error_reason),
        }
        for model_name, checkpoint_path in {
            "Primary Baseline (FP16)": baseline_checkpoint_path(),
            "FP32 Research Reference": baseline_checkpoint_path(),
            "Frozen Checkpoint (FP32)": frozen_checkpoint_path(),
        }.items():
            if checkpoint_path.exists():
                results["models"][model_name]["model_file_size_mb"] = round(
                    checkpoint_path.stat().st_size / (1024**2),
                    2,
                )
                results["models"][model_name]["checkpoint"] = relative_checkpoint_label(checkpoint_path)
    else:
        results["models"] = {
            "Primary Baseline (FP16)": evaluate_checkpoint(
                baseline_checkpoint_path(),
                dtype_label="fp16",
                use_half=True,
            ),
            "FP32 Research Reference": evaluate_checkpoint(
                baseline_checkpoint_path(),
                dtype_label="fp32",
                use_half=False,
            ),
            "Frozen Checkpoint (FP32)": evaluate_checkpoint(
                frozen_checkpoint_path(),
                dtype_label="fp32",
                use_half=False,
            ),
        }

    save_reference_model_results(results)

    for model_name, entry in results["models"].items():
        print(f"\n{model_name}")
        print(f"  status : {entry['status']}")
        if entry["status"] != "measured":
            print(f"  reason : {entry['reason']}")
            continue
        print(f"  dtype  : {entry.get('dtype', 'unknown')}")
        print(f"  device : {entry.get('device', 'unknown')}")
        print(f"  acc    : {entry['accuracy']:.4f}")
        print(f"  f1     : {entry['f1']:.4f}")
        print(f"  lat    : {entry['latency_ms']:.2f} ms")
        print(f"  size   : {entry['model_file_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
