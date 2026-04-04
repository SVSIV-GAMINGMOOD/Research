from __future__ import annotations

import json
from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.MRPC.common import compute_fp16_baseline_onnx_metrics, evaluate_onnx_model, load_mrpc_dataset, total_onnx_size_mb
from GLUE.MRPC.task_config import (
    EVALUATION_SETTINGS,
    ONNX_MODEL_SPECS,
    all_model_results_path,
    all_model_results_txt_path,
    save_all_model_results,
)


def main() -> None:
    dataset = load_mrpc_dataset(format_type="numpy", max_samples=EVALUATION_SETTINGS["max_samples"])
    all_results = {}

    for method, spec in ONNX_MODEL_SPECS.items():
        model_path = spec["path"]
        print(f"{'=' * 60}")
        print(f"Evaluating: {method}")
        print(f"File      : {model_path}")
        if not model_path.exists():
            print("  skipped - file not found")
            all_results[method] = None
            continue

        metrics = evaluate_onnx_model(
            model_path,
            dataset,
            batch_size=EVALUATION_SETTINGS["batch_size"],
            latency_runs=EVALUATION_SETTINGS["latency_runs"],
            threads=EVALUATION_SETTINGS["threads"],
            providers=spec.get("providers"),
            required_provider=spec.get("required_provider"),
        )
        metrics["onnx_size_mb"] = total_onnx_size_mb(model_path)
        all_results[method] = {
            "accuracy": round(metrics["accuracy"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "latency_avg": round(metrics["latency_avg_ms"], 2),
            "latency_p50": round(metrics["latency_p50_ms"], 2),
            "latency_p95": round(metrics["latency_p95_ms"], 2),
            "onnx_mb": round(metrics["onnx_size_mb"], 2),
        }
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Latency  : {metrics['latency_avg_ms']:.2f} ms")
        print(f"  ONNX Size: {metrics['onnx_size_mb']:.2f} MB")

    if "Primary Baseline (FP16)" in all_results and all_results["Primary Baseline (FP16)"] is not None:
        compute_fp16_baseline_onnx_metrics(overwrite=True)

    print(f"\n{'=' * 72}")
    print("MRPC RESULTS - ALL MODELS")
    print(f"{'=' * 72}")
    print(f"  {'Method':<28} {'Acc':>7} {'F1':>7} {'Latency':>10} {'ONNX MB':>10}")
    print("  " + "-" * 65)
    for method, metrics in all_results.items():
        if metrics is None:
            print(f"  {method:<28}  -  (file missing)")
            continue
        print(
            f"  {method:<28} {metrics['accuracy']:>7.4f} {metrics['f1']:>7.4f} "
            f"{metrics['latency_avg']:>8.2f} ms {metrics['onnx_mb']:>8.2f} MB"
        )

    save_all_model_results(all_results)

    lines = [f"{'Method':<28} {'Accuracy':>9} {'F1':>7} {'Latency(ms)':>12} {'ONNX(MB)':>10}", "-" * 70]
    for method, metrics in all_results.items():
        if metrics is None:
            lines.append(f"{method:<28}  - (missing)")
            continue
        lines.append(
            f"{method:<28} {metrics['accuracy']:>9.4f} {metrics['f1']:>7.4f} "
            f"{metrics['latency_avg']:>11.2f} {metrics['onnx_mb']:>10.2f}"
        )
    all_model_results_txt_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {all_model_results_path()}")
    print(f"Saved: {all_model_results_txt_path()}")


if __name__ == "__main__":
    main()

