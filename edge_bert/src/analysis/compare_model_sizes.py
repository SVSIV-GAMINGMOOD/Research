"""
=============================================================
STAGE 6B — Effective Size Comparison (ALL Models)
=============================================================
Uses BOTH:
  1. Actual ONNX file sizes  (os.path.getsize)
  2. Theoretical effective bit-count sizes

Models covered:
  baseline_fp32.onnx
  baseline_int8.onnx
  frozen_fp32.onnx
  frozen_int8.onnx
  greedy_quant_model.onnx
  mixed_precision_real_quant.onnx   ← SA v1
  hybrid_quant_model.onnx           ← SA + INT8 embeddings (Ours)

Run:
  python step3_size_comparison.py
Output:
  results/size_comparison.json
  results/size_comparison.txt
=============================================================
"""

import os
import json
import torch
from pathlib import Path
from transformers import DistilBertForSequenceClassification

# ----------------------------------------------------------
# PATHS — adjust if your models/ folder is elsewhere
# ----------------------------------------------------------
SRC_ROOT    = Path(__file__).resolve().parents[1]
MODELS_DIR  = SRC_ROOT / "models"
RESULTS_DIR = SRC_ROOT / "results"
MODEL_NAME  = "distilbert-base-uncased"
os.makedirs(RESULTS_DIR, exist_ok=True)

ONNX_FILES = {
    "FP32 Baseline":        "baseline_fp32.onnx",
    "INT8 Uniform":         "baseline_int8.onnx",
    "FAR Frozen FP32":      "frozen_fp32.onnx",
    "FAR Frozen INT8":      "frozen_int8.onnx",
    "Greedy Mixed":         "greedy_quant_model.onnx",
    "SA Mixed (v1)":        "mixed_precision_real_quant.onnx",
    "Hybrid INT8+SA (Ours)":"hybrid_quant_model.onnx",
}

# ----------------------------------------------------------
# SA CONFIGURATION (from your Stage 4/6 search)
# ----------------------------------------------------------
SA_CONFIG = {
    "distilbert.transformer.layer.0.attention.q_lin":   8,
    "distilbert.transformer.layer.0.attention.k_lin":   6,
    "distilbert.transformer.layer.0.attention.v_lin":   4,
    "distilbert.transformer.layer.0.attention.out_lin": 8,
    "distilbert.transformer.layer.0.ffn.lin1":          4,
    "distilbert.transformer.layer.0.ffn.lin2":          6,
    "distilbert.transformer.layer.1.attention.q_lin":   6,
    "distilbert.transformer.layer.1.attention.k_lin":   4,
    "distilbert.transformer.layer.1.attention.v_lin":   4,
    "distilbert.transformer.layer.1.attention.out_lin": 6,
    "distilbert.transformer.layer.2.attention.q_lin":   4,
    "distilbert.transformer.layer.2.attention.k_lin":   6,
    "distilbert.transformer.layer.2.attention.v_lin":   4,
    "distilbert.transformer.layer.2.attention.out_lin": 6,
    "distilbert.transformer.layer.3.attention.q_lin":   8,
    "distilbert.transformer.layer.3.attention.k_lin":   6,
    "distilbert.transformer.layer.3.attention.v_lin":   4,
    "distilbert.transformer.layer.3.attention.out_lin": 8,
    "distilbert.transformer.layer.3.ffn.lin1":          8,
    "distilbert.transformer.layer.3.ffn.lin2":          6,
    "distilbert.transformer.layer.4.attention.q_lin":   8,
    "distilbert.transformer.layer.4.attention.k_lin":   8,
    "distilbert.transformer.layer.4.attention.v_lin":   8,
    "distilbert.transformer.layer.4.attention.out_lin": 4,
    "distilbert.transformer.layer.4.ffn.lin1":          4,
    "distilbert.transformer.layer.4.ffn.lin2":          6,
    "distilbert.transformer.layer.5.attention.q_lin":   4,
    "distilbert.transformer.layer.5.attention.k_lin":   4,
    "distilbert.transformer.layer.5.attention.v_lin":   8,
    "distilbert.transformer.layer.5.attention.out_lin": 6,
    "distilbert.transformer.layer.5.ffn.lin1":          6,
    "distilbert.transformer.layer.5.ffn.lin2":          4,
    "pre_classifier": 4,
    "classifier":     8,
}
GREEDY_CONFIG = {
    "distilbert.transformer.layer.0.ffn.lin2":          8,
    "distilbert.transformer.layer.3.ffn.lin2":          8,
    "distilbert.transformer.layer.3.attention.out_lin": 8,
    "distilbert.transformer.layer.3.ffn.lin1":          8,
    "distilbert.transformer.layer.4.ffn.lin2":          8,
    "distilbert.transformer.layer.5.attention.q_lin":   8,
    "distilbert.transformer.layer.1.attention.k_lin":   8,
    "distilbert.transformer.layer.1.attention.v_lin":   8,
    "distilbert.transformer.layer.4.attention.k_lin":   8,
    "distilbert.transformer.layer.4.attention.v_lin":   8,
    "distilbert.transformer.layer.5.ffn.lin2":          8,
    "pre_classifier":                                   6,
    "distilbert.transformer.layer.0.attention.v_lin":   6,
    "distilbert.transformer.layer.3.attention.q_lin":   6,
    "distilbert.transformer.layer.4.attention.out_lin": 6,
    "distilbert.transformer.layer.4.ffn.lin1":          6,
    "classifier":                                       6,
    "distilbert.transformer.layer.0.attention.k_lin":   6,
    "distilbert.transformer.layer.0.attention.out_lin": 6,
    "distilbert.transformer.layer.2.attention.k_lin":   6,
    "distilbert.transformer.layer.3.attention.v_lin":   6,
    "distilbert.transformer.layer.5.attention.k_lin":   6,
    "distilbert.transformer.layer.5.attention.out_lin": 4,
    "distilbert.transformer.layer.0.attention.q_lin":   4,
    "distilbert.transformer.layer.1.attention.out_lin": 4,
    "distilbert.transformer.layer.2.attention.q_lin":   4,
    "distilbert.transformer.layer.2.attention.out_lin": 4,
    "distilbert.transformer.layer.4.attention.q_lin":   4,
    "distilbert.transformer.layer.5.attention.v_lin":   4,
    "distilbert.transformer.layer.5.ffn.lin1":          4,
    "distilbert.transformer.layer.1.attention.q_lin":   4,
    "distilbert.transformer.layer.3.attention.k_lin":   4,
    "distilbert.transformer.layer.0.ffn.lin1":          4,
    "distilbert.transformer.layer.2.attention.v_lin":   4,
    "distilbert.transformer.layer.1.ffn.lin1":          8,
    "distilbert.transformer.layer.2.ffn.lin2":          8,
    "distilbert.transformer.layer.2.ffn.lin1":          8,
    "distilbert.transformer.layer.1.ffn.lin2":          8,
}
LOCKED_LAYERS = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1",
}
for l in LOCKED_LAYERS:
    SA_CONFIG[l] = 8

# ----------------------------------------------------------
# THEORETICAL BIT-COUNT SIZE
# ----------------------------------------------------------
def is_embedding_param(name):
    return "embeddings" in name and "LayerNorm" not in name

def get_linear_name(param_name):
    return param_name.replace(".weight", "").replace(".bias", "")

def compute_theoretical_size(model, linear_config=None,
                              embed_bits=32, default_bits=32):
    total_bits, total_params = 0, 0
    for pname, param in model.named_parameters():
        n     = param.numel()
        lname = get_linear_name(pname)
        total_params += n
        if is_embedding_param(pname):
            total_bits += n * embed_bits
        elif linear_config and lname in linear_config:
            total_bits += n * linear_config[lname]
        else:
            total_bits += n * default_bits
    return {
        "theoretical_mb": round(total_bits / 8 / 1024 / 1024, 2),
        "eff_bits":        round(total_bits / total_params, 2),
        "total_params":    total_params,
    }

# Theoretical configs per method
THEORETICAL_CONFIGS = {
    "FP32 Baseline":         dict(default_bits=32),
    "INT8 Uniform":          dict(default_bits=8),
    "FAR Frozen FP32":       dict(default_bits=32),   # same params, just some frozen
    "FAR Frozen INT8":       dict(default_bits=8),
    "Greedy Mixed":          dict(linear_config=GREEDY_CONFIG),
    "SA Mixed (v1)":         dict(linear_config=SA_CONFIG),
    "Hybrid INT8+SA (Ours)": dict(linear_config=SA_CONFIG, embed_bits=8),
}

# ----------------------------------------------------------
# LOAD MODEL FOR THEORETICAL CALC
# ----------------------------------------------------------
print("Loading DistilBERT parameter map...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}\n")

fp32_theoretical = compute_theoretical_size(model)["theoretical_mb"]

# ----------------------------------------------------------
# COLLECT ALL RESULTS
# ----------------------------------------------------------
results = {}

print("=" * 72)
print(f"  {'Method':<28} {'ONNX File':>12} {'Theoretical':>13} {'Eff.Bits':>10} {'Ratio':>8}")
print("  " + "-" * 70)

for method, fname in ONNX_FILES.items():
    fpath = MODELS_DIR / fname

    # ONNX actual file size
    if os.path.exists(fpath):
        onnx_mb = os.path.getsize(fpath) / 1024 / 1024
    else:
        onnx_mb = None
        print(f"  ⚠  NOT FOUND: {fpath}")

    # Theoretical size
    cfg  = THEORETICAL_CONFIGS.get(method, {})
    theo = compute_theoretical_size(model, **cfg)

    ratio = fp32_theoretical / theo["theoretical_mb"]

    results[method] = {
        "onnx_file":      fname,
        "onnx_mb":        round(onnx_mb, 2) if onnx_mb else None,
        "theoretical_mb": theo["theoretical_mb"],
        "eff_bits":        theo["eff_bits"],
        "total_params":    theo["total_params"],
        "compression_ratio": round(ratio, 2),
    }

    onnx_s = f"{onnx_mb:.2f} MB" if onnx_mb else "  N/A  "
    print(f"  {method:<28} {onnx_s:>12} "
          f"{theo['theoretical_mb']:>10.2f} MB "
          f"{theo['eff_bits']:>9.2f}b "
          f"{ratio:>7.2f}×")

print("=" * 72)
print(f"\n  Note: ONNX file size includes graph metadata overhead.")
print(f"  Theoretical size = pure weight storage (used in paper).")

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
json_path = RESULTS_DIR / "size_comparison.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

txt_path = RESULTS_DIR / "size_comparison.txt"
with open(txt_path, "w") as f:
    f.write(f"{'Method':<28} {'ONNX MB':>10} {'Theory MB':>11} {'Eff Bits':>10} {'Ratio':>8}\n")
    f.write("-" * 70 + "\n")
    for method, r in results.items():
        onnx_s = f"{r['onnx_mb']:.2f}" if r["onnx_mb"] else "N/A"
        f.write(f"{method:<28} {onnx_s:>10} "
                f"{r['theoretical_mb']:>10.2f} "
                f"{r['eff_bits']:>10.2f} "
                f"{r['compression_ratio']:>8.2f}×\n")

print(f"\n  Saved: {json_path}")
print(f"  Saved: {txt_path}")
