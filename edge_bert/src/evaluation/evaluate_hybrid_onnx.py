from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, RESULTS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


MODEL_PATH = MODELS_DIR / "hybrid_quant_model.onnx"
MAX_SAMPLES = 872
LATENCY_RUNS = 100
THREADS = 4

PREVIOUS_RESULTS = {
    "FP32 Baseline": {"accuracy": 0.9117, "f1": None, "size_mb": 255.0, "latency_ms": None},
    "INT8 Uniform": {"accuracy": 0.9117, "f1": None, "size_mb": 63.75, "latency_ms": None},
    "Greedy Mixed": {"accuracy": 0.9106, "f1": 0.9124, "size_mb": None, "latency_ms": 14.45},
    "SA Mixed (v1)": {"accuracy": 0.9002, "f1": 0.9037, "size_mb": None, "latency_ms": 14.41},
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_sst2_validation_dataset(format_type="numpy", max_samples=MAX_SAMPLES)
    metrics = evaluate_onnx_model(
        MODEL_PATH,
        dataset,
        batch_size=32,
        latency_runs=LATENCY_RUNS,
        threads=THREADS,
    )

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
    for method, result in PREVIOUS_RESULTS.items():
        acc_str = f"{result['accuracy']:.4f}"
        f1_str = f"{result['f1']:.4f}" if result["f1"] is not None else "   -   "
        size_str = f"{result['size_mb']:.1f} MB" if result["size_mb"] is not None else "   -   "
        lat_str = f"{result['latency_ms']:.2f} ms" if result["latency_ms"] is not None else "   -   "
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
            "samples": MAX_SAMPLES,
            "batch_size": 32,
            "latency_runs": LATENCY_RUNS,
            "threads": THREADS,
        },
    }

    json_path = RESULTS_DIR / "hybrid_eval_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {json_path}")

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
        + "\n"
    )
    print(f"  Results saved: {txt_path}")


if __name__ == "__main__":
    main()
