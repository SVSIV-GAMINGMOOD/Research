from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import ensure_cuda_available, evaluate_pytorch_model, load_classifier_checkpoint, load_mrpc_dataset
from GLUE.MRPC.task_config import EVALUATION_SETTINGS, baseline_checkpoint_path, save_reference_model_results


def evaluate_checkpoint(*, use_half: bool) -> dict[str, object]:
    if use_half:
        ensure_cuda_available("MRPC FP16 checkpoint evaluation")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_mrpc_dataset(format_type="torch", max_samples=EVALUATION_SETTINGS["max_samples"])
    model = load_classifier_checkpoint(
        baseline_checkpoint_path(),
        device=device,
        use_half=use_half,
    )
    metrics = evaluate_pytorch_model(
        model,
        dataset,
        latency_runs=EVALUATION_SETTINGS["latency_runs"],
    )
    checkpoint_path = baseline_checkpoint_path()
    return {
        "status": "measured",
        "backend": "pytorch",
        "dtype": "fp16" if use_half else "fp32",
        "checkpoint": str(checkpoint_path.relative_to(checkpoint_path.parents[2])).replace("\\", "/"),
        "device": device,
        "accuracy": round(metrics["accuracy"], 4),
        "f1": round(metrics["f1"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "latency_ms": round(metrics["latency_ms"], 2),
        "model_file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRPC checkpoint reference models.")
    parser.add_argument("--prepare-only", action="store_true", help="Write placeholder entries without evaluation.")
    args = parser.parse_args()

    results = {
        "strategy": {
            "primary_baseline_precision": "fp16",
            "reference_precision": "fp32",
            "notes": "MRPC uses FP16 as the primary baseline and FP32 as the research reference.",
        },
        "models": {},
    }

    if args.prepare_only:
        checkpoint_path = baseline_checkpoint_path()
        size_mb = round(checkpoint_path.stat().st_size / (1024**2), 2) if checkpoint_path.exists() else None
        for label, dtype in (
            ("Primary Baseline (FP16)", "fp16"),
            ("FP32 Research Reference", "fp32"),
        ):
            results["models"][label] = {
                "status": "pending_manual_run",
                "backend": "pytorch",
                "dtype": dtype,
                "checkpoint": str(checkpoint_path.relative_to(checkpoint_path.parents[2])).replace("\\", "/"),
                "accuracy": None,
                "f1": None,
                "roc_auc": None,
                "latency_ms": None,
                "model_file_size_mb": size_mb,
            }
    else:
        results["models"] = {
            "Primary Baseline (FP16)": evaluate_checkpoint(use_half=True),
            "FP32 Research Reference": evaluate_checkpoint(use_half=False),
        }

    save_reference_model_results(results)

    for model_name, entry in results["models"].items():
        print(f"\n{model_name}")
        print(f"  status : {entry['status']}")
        if entry["status"] != "measured":
            continue
        print(f"  dtype  : {entry['dtype']}")
        print(f"  device : {entry['device']}")
        print(f"  acc    : {entry['accuracy']:.4f}")
        print(f"  f1     : {entry['f1']:.4f}")
        print(f"  rocauc : {entry['roc_auc']:.4f}")
        print(f"  lat    : {entry['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()

