from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.model_workflows import MODELS_DIR, RESULTS_DIR, evaluate_onnx_model, load_sst2_validation_dataset


ONNX_FILES = {
    "FP32 Baseline": MODELS_DIR / "baseline_fp32.onnx",
    "INT8 Uniform": MODELS_DIR / "baseline_int8.onnx",
    "FAR Frozen FP32": MODELS_DIR / "frozen_fp32.onnx",
    "FAR Frozen INT8": MODELS_DIR / "frozen_int8.onnx",
    "Greedy Mixed": MODELS_DIR / "greedy_quant_model.onnx",
    "SA Mixed (v1)": MODELS_DIR / "mixed_precision_real_quant.onnx",
    "Hybrid INT8+SA (Ours)": MODELS_DIR / "hybrid_quant_model.onnx",
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_sst2_validation_dataset(format_type="numpy", max_samples=872)
    all_results = {}

    for method, model_path in ONNX_FILES.items():
        print(f"{'=' * 60}")
        print(f"Evaluating: {method}")
        print(f"File      : {model_path}")
        if not model_path.exists():
            print("  skipped - file not found")
            all_results[method] = None
            continue

        metrics = evaluate_onnx_model(model_path, dataset, latency_runs=100, threads=4)
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
        print(f"  Latency  : {metrics['latency_avg_ms']:.2f} ms  (p95: {metrics['latency_p95_ms']:.2f} ms)")
        print(f"  ONNX Size: {metrics['onnx_size_mb']:.2f} MB")

    print(f"\n{'=' * 72}")
    print("FULL RESULTS - ALL MODELS")
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
    print(f"{'=' * 72}")

    json_path = RESULTS_DIR / "all_model_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    txt_lines = [f"{'Method':<28} {'Accuracy':>9} {'F1':>7} {'Latency(ms)':>12} {'ONNX(MB)':>10}", "-" * 70]
    for method, metrics in all_results.items():
        if metrics is None:
            txt_lines.append(f"{method:<28}  - (missing)")
            continue
        txt_lines.append(
            f"{method:<28} {metrics['accuracy']:>9.4f} {metrics['f1']:>7.4f} "
            f"{metrics['latency_avg']:>11.2f} {metrics['onnx_mb']:>10.2f}"
        )

    txt_path = RESULTS_DIR / "all_model_results.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n")
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()
