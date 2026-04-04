from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import RESULTS_DIR


METHOD_ORDER = [
    "FP32 Baseline",
    "INT8 Uniform",
    "FAR Frozen FP32",
    "FAR Frozen INT8",
    "Greedy Mixed",
    "SA (v1)",
    "Hybrid SA (ours)",
]


def main() -> None:
    results_path = RESULTS_DIR / "all_model_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing ONNX comparison results: {results_path}")

    all_results = json.loads(results_path.read_text(encoding="utf-8"))

    print("\n===== ONNX MODEL COMPARISON =====\n")
    print(f"{'Method':<28} {'Acc':>7} {'F1':>7} {'Latency':>10} {'ONNX MB':>10}")
    print("-" * 68)
    for method in METHOD_ORDER:
        metrics = all_results.get(method)
        if not metrics:
            print(f"{method:<28} {'-':>7} {'-':>7} {'-':>10} {'-':>10}")
            continue
        print(
            f"{method:<28} {metrics['accuracy']:>7.4f} {metrics['f1']:>7.4f} "
            f"{metrics['latency_avg']:>8.2f} ms {metrics['onnx_mb']:>8.2f}"
        )

    print(f"\nSource: {results_path}")


if __name__ == "__main__":
    main()

