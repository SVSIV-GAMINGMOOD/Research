from __future__ import annotations

import json
from pathlib import Path
import sys

from transformers import DistilBertForSequenceClassification


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.QQP.common import total_onnx_size_mb
from GLUE.QQP.task_config import (
    MODEL_NAME,
    NUM_LABELS,
    QUANTIZATION_SETTINGS,
    RESULTS_DIR,
    baseline_fp16_onnx_path,
    baseline_fp32_onnx_path,
    baseline_int8_onnx_path,
    get_locked_layers,
    greedy_onnx_path,
    hybrid_onnx_path,
    load_greedy_config,
    load_sa_best_config,
    save_json,
    sa_onnx_path,
)


ONNX_PATHS = {
    "FP32 Research Reference": baseline_fp32_onnx_path(),
    "Primary Baseline (FP16)": baseline_fp16_onnx_path(),
    "INT8 Uniform": baseline_int8_onnx_path(),
    "Greedy Mixed": greedy_onnx_path(),
    "SA (v1)": sa_onnx_path(),
    "Hybrid SA (ours)": hybrid_onnx_path(),
}


def is_embedding_param(name: str) -> bool:
    return "embeddings" in name and "LayerNorm" not in name


def get_linear_name(param_name: str) -> str:
    return param_name.replace(".weight", "").replace(".bias", "")


def compute_theoretical_size(
    model,
    *,
    linear_config: dict[str, int] | None = None,
    embed_bits: int = 32,
    default_bits: int = 32,
) -> dict[str, float]:
    total_bits = 0
    total_params = 0
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
        "effective_bits": round(total_bits / total_params, 2),
        "total_params": total_params,
    }


def main() -> None:
    greedy_config = load_greedy_config(required=False)
    sa_config = load_sa_best_config(required=False)
    for layer in get_locked_layers():
        sa_config[layer] = 8

    theoretical_configs = {
        "FP32 Research Reference": {"default_bits": 32, "embed_bits": 32},
        "Primary Baseline (FP16)": {"default_bits": 16, "embed_bits": 16},
        "INT8 Uniform": {"default_bits": 8, "embed_bits": 8},
        "Greedy Mixed": {"linear_config": greedy_config, "embed_bits": 32, "default_bits": 32},
        "SA (v1)": {"linear_config": sa_config, "embed_bits": 32, "default_bits": 32},
        "Hybrid SA (ours)": {
            "linear_config": sa_config,
            "embed_bits": QUANTIZATION_SETTINGS["embedding_bits"],
            "default_bits": 32,
        },
    }

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.eval()

    fp32_reference = compute_theoretical_size(model)["theoretical_mb"]
    results = {}

    print("=" * 76)
    print(f"  {'Method':<28} {'ONNX MB':>10} {'Theory MB':>11} {'Eff Bits':>10} {'Ratio':>8}")
    print("  " + "-" * 72)

    for method, model_path in ONNX_PATHS.items():
        theoretical = compute_theoretical_size(model, **theoretical_configs[method])
        ratio = fp32_reference / theoretical["theoretical_mb"]
        onnx_mb = round(total_onnx_size_mb(model_path), 2) if model_path.exists() else None
        results[method] = {
            "onnx_file": str(model_path.name),
            "onnx_mb": onnx_mb,
            "theoretical_mb": theoretical["theoretical_mb"],
            "effective_bits": theoretical["effective_bits"],
            "total_params": theoretical["total_params"],
            "compression_ratio": round(ratio, 2),
        }
        onnx_label = f"{onnx_mb:.2f}" if onnx_mb is not None else "N/A"
        print(
            f"  {method:<28} {onnx_label:>10} "
            f"{theoretical['theoretical_mb']:>11.2f} {theoretical['effective_bits']:>10.2f} {ratio:>8.2f}x"
        )

    json_path = RESULTS_DIR / "size_comparison.json"
    txt_path = RESULTS_DIR / "size_comparison.txt"
    save_json(json_path, results)

    lines = [f"{'Method':<28} {'ONNX MB':>10} {'Theory MB':>11} {'Eff Bits':>10} {'Ratio':>8}", "-" * 72]
    for method, payload in results.items():
        onnx_value = f"{payload['onnx_mb']:.2f}" if payload["onnx_mb"] is not None else "N/A"
        lines.append(
            f"{method:<28} {onnx_value:>10} {payload['theoretical_mb']:>11.2f} "
            f"{payload['effective_bits']:>10.2f} {payload['compression_ratio']:>7.2f}x"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
