from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import EVALUATION_SETTINGS, load_all_model_results
from shared.model_workflows import MODELS_DIR, RESULTS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


MODEL_PATH = MODELS_DIR / "hybrid_quant_model.onnx"
COMPARISON_METHODS = ["FP32 Baseline", "INT8 Uniform", "Greedy Mixed", "SA (v1)"]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_sst2_validation_dataset(
        format_type="numpy",
        max_samples=EVALUATION_SETTINGS["max_samples"],
    )
    metrics = evaluate_onnx_model(
        MODEL_PATH,
        dataset,
        batch_size=EVALUATION_SETTINGS["batch_size"],
        latency_runs=EVALUATION_SETTINGS["latency_runs"],
        threads=EVALUATION_SETTINGS["threads"],
    )
    previous_results = load_all_model_results()

    print("\n" + "=" * 60)
    print("HYBRID QUANTIZATION - EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  F1 Score         : {metrics['f1']:.4f}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  ONNX File Size   : {metrics['onnx_size_mb']:.2f} MB")
    print(f"  Avg Latency      : {metrics['latency_avg_ms']:.2f} ms")
    print(f"  P50 Latency      : {metrics['latency_p50_ms']:.2f} ms")
    print(f"  P95 Latency      : {metrics['latency_p95_ms']:.2f} ms")
    print(f"  P99 Latency      : {metrics['latency_p99_ms']:.2f} ms")
    print("=" * 60)

    print("\n  FULL COMPARISON TABLE (paper-ready)")
    print(f"  {'Method':<25} {'Accuracy':>10} {'F1':>8} {'Size':>10} {'Latency':>12}")
    print("  " + "-" * 70)
    for method in COMPARISON_METHODS:
        result = previous_results.get(method, {})
        acc_str = f"{result['accuracy']:.4f}" if result.get("accuracy") is not None else "   -   "
        f1_str = f"{result['f1']:.4f}" if result.get("f1") is not None else "   -   "
        size_str = f"{result['onnx_mb']:.1f} MB" if result.get("onnx_mb") is not None else "   -   "
        lat_str = f"{result['latency_avg']:.2f} ms" if result.get("latency_avg") is not None else "   -   "
        print(f"  {method:<25} {acc_str:>10} {f1_str:>8} {size_str:>10} {lat_str:>12}")

    print(
        f"  {'Hybrid (INT8 Emb+SA)':<25} {metrics['accuracy']:>10.4f} {metrics['f1']:>8.4f} "
        f"{metrics['onnx_size_mb']:>8.1f} MB {metrics['latency_avg_ms']:>10.2f} ms  <- NEW"
    )
    print("  " + "-" * 70)

    results = {
        "model": str(MODEL_PATH),
        "metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "onnx_size_mb": round(metrics["onnx_size_mb"], 2),
        },
        "latency_ms": {
            "avg": round(metrics["latency_avg_ms"], 2),
            "p50": round(metrics["latency_p50_ms"], 2),
            "p95": round(metrics["latency_p95_ms"], 2),
            "p99": round(metrics["latency_p99_ms"], 2),
        },
        "eval_config": {
            "samples": EVALUATION_SETTINGS["max_samples"],
            "batch_size": EVALUATION_SETTINGS["batch_size"],
            "latency_runs": EVALUATION_SETTINGS["latency_runs"],
            "threads": EVALUATION_SETTINGS["threads"],
        },
    }
    json_path = RESULTS_DIR / "hybrid_eval_results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    txt_path = RESULTS_DIR / "hybrid_eval_results.txt"
    txt_path.write_text(
        "\n".join(
            [
                "HYBRID QUANTIZATION - EVALUATION RESULTS",
                "=" * 60,
                f"Accuracy         : {metrics['accuracy']:.4f}",
                f"F1 Score         : {metrics['f1']:.4f}",
                f"ROC-AUC          : {metrics['roc_auc']:.4f}",
                f"ONNX File Size   : {metrics['onnx_size_mb']:.2f} MB",
                f"Avg Latency      : {metrics['latency_avg_ms']:.2f} ms",
                f"P95 Latency      : {metrics['latency_p95_ms']:.2f} ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  Results saved: {json_path}")
    print(f"  Results saved: {txt_path}")


if __name__ == "__main__":
    main()

