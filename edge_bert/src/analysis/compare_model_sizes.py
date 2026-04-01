import json
import os
from pathlib import Path
import sys

import torch
from transformers import DistilBertForSequenceClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_BITS,
    DEFAULT_EMBEDDING_BITS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
    get_locked_layers,
    load_greedy_config,
    load_sa_best_config,
)


ONNX_FILES = {
    "FP32 Baseline": "baseline_fp32.onnx",
    "INT8 Uniform": "baseline_int8.onnx",
    "FAR Frozen FP32": "frozen_fp32.onnx",
    "FAR Frozen INT8": "frozen_int8.onnx",
    "Greedy Mixed": "greedy_quant_model.onnx",
    "SA Mixed (v1)": "mixed_precision_real_quant.onnx",
    "Hybrid INT8+SA (Ours)": "hybrid_quant_model.onnx",
}

SA_CONFIG = load_sa_best_config()
GREEDY_CONFIG = load_greedy_config()
LOCKED_LAYERS = set(get_locked_layers())
for layer in LOCKED_LAYERS:
    SA_CONFIG[layer] = 8


def is_embedding_param(name: str) -> bool:
    return "embeddings" in name and "LayerNorm" not in name


def get_linear_name(param_name: str) -> str:
    return param_name.replace(".weight", "").replace(".bias", "")


def compute_theoretical_size(model, linear_config=None, embed_bits=DEFAULT_BITS, default_bits=DEFAULT_BITS):
    total_bits, total_params = 0, 0
    for param_name, param in model.named_parameters():
        params = param.numel()
        layer_name = get_linear_name(param_name)
        total_params += params
        if is_embedding_param(param_name):
            total_bits += params * embed_bits
        elif linear_config and layer_name in linear_config:
            total_bits += params * linear_config[layer_name]
        else:
            total_bits += params * default_bits
    return {
        "theoretical_mb": round(total_bits / 8 / 1024 / 1024, 2),
        "eff_bits": round(total_bits / total_params, 2),
        "total_params": total_params,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    theoretical_configs = {
        "FP32 Baseline": dict(default_bits=32),
        "INT8 Uniform": dict(default_bits=8),
        "FAR Frozen FP32": dict(default_bits=32),
        "FAR Frozen INT8": dict(default_bits=8),
        "Greedy Mixed": dict(linear_config=GREEDY_CONFIG),
        "SA Mixed (v1)": dict(linear_config=SA_CONFIG),
        "Hybrid INT8+SA (Ours)": dict(linear_config=SA_CONFIG, embed_bits=DEFAULT_EMBEDDING_BITS),
    }

    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )
    model.eval()
    fp32_theoretical = compute_theoretical_size(model)["theoretical_mb"]

    results = {}
    print("=" * 72)
    print(f"  {'Method':<28} {'ONNX File':>12} {'Theoretical':>13} {'Eff.Bits':>10} {'Ratio':>8}")
    print("  " + "-" * 70)

    for method, file_name in ONNX_FILES.items():
        model_path = MODELS_DIR / file_name
        onnx_mb = os.path.getsize(model_path) / 1024 / 1024 if model_path.exists() else None
        theoretical = compute_theoretical_size(model, **theoretical_configs.get(method, {}))
        ratio = fp32_theoretical / theoretical["theoretical_mb"]
        results[method] = {
            "onnx_file": file_name,
            "onnx_mb": round(onnx_mb, 2) if onnx_mb is not None else None,
            "theoretical_mb": theoretical["theoretical_mb"],
            "eff_bits": theoretical["eff_bits"],
            "total_params": theoretical["total_params"],
            "compression_ratio": round(ratio, 2),
        }
        onnx_str = f"{onnx_mb:.2f} MB" if onnx_mb is not None else "  N/A  "
        print(
            f"  {method:<28} {onnx_str:>12} "
            f"{theoretical['theoretical_mb']:>10.2f} MB "
            f"{theoretical['eff_bits']:>9.2f}b "
            f"{ratio:>7.2f}x"
        )

    json_path = RESULTS_DIR / "size_comparison.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    txt_path = RESULTS_DIR / "size_comparison.txt"
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(f"{'Method':<28} {'ONNX MB':>10} {'Theory MB':>11} {'Eff Bits':>10} {'Ratio':>8}\n")
        file.write("-" * 70 + "\n")
        for method, result in results.items():
            onnx_str = f"{result['onnx_mb']:.2f}" if result["onnx_mb"] is not None else "N/A"
            file.write(
                f"{method:<28} {onnx_str:>10} {result['theoretical_mb']:>10.2f} "
                f"{result['eff_bits']:>10.2f} {result['compression_ratio']:>8.2f}x\n"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
