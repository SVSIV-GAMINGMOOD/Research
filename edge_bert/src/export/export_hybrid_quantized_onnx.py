"""
=============================================================
STAGE 6B — Hybrid Quantization Export
=============================================================
Strategy (aligned with NVFP4 philosophy):
  Embeddings       → INT8  (fixed, like FP8 scaling in NVFP4)
  LayerNorm/Softmax→ FP32  (always, too sensitive)
  Locked layers    → 8-bit (FAR sensitivity identified)
  Remaining linear → SA mixed precision {4, 6, 8}

Run:
  python step1_hybrid_export.py
Output:
  models/hybrid_quant_model.onnx
  models/hybrid_config_summary.json
=============================================================
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from transformers import DistilBertForSequenceClassification

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
DEVICE         = "cpu"   # Force CPU (edge simulation)
MODEL_NAME     = "distilbert-base-uncased"
SRC_ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR     = SRC_ROOT / "models"
BASELINE_PATH  = MODELS_DIR / "baseline_best.pt"
ONNX_OUT       = MODELS_DIR / "hybrid_quant_model.onnx"
SUMMARY_OUT    = MODELS_DIR / "hybrid_config_summary.json"
SEQ_LEN        = 128

# ----------------------------------------------------------
# SA BEST CONFIGURATION (from your Stage 4/6 search)
# These are the bit-widths SA found for transformer linears
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

# FAR-identified sensitive layers — always locked to 8-bit
LOCKED_LAYERS = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1",
}

# Enforce locked layers override SA config
for layer in LOCKED_LAYERS:
    SA_CONFIG[layer] = 8

# ----------------------------------------------------------
# QUANTIZED LINEAR LAYER (true weight quantization)
# ----------------------------------------------------------
class QuantizedLinear(nn.Module):
    """
    Simulates mixed-precision quantization.
    Stores weights as INT32 quantized values with scale/zero-point.
    Dequantizes on the fly during forward pass.
    Same pattern as NVFP4 micro-block dequantize-before-MAC.
    """
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear_layer.weight.data

        q_min = 0
        q_max = 2 ** bits - 1
        w_min = weight.min()
        w_max = weight.max()

        scale      = (w_max - w_min) / (q_max - q_min + 1e-8)
        zero_point = torch.round(q_min - w_min / (scale + 1e-8))

        q_weight = torch.clamp(
            torch.round(weight / scale + zero_point), q_min, q_max
        ).to(torch.int32)

        self.register_buffer("q_weight",    q_weight)
        self.register_buffer("scale",       scale)
        self.register_buffer("zero_point",  zero_point)

        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return nn.functional.linear(x, weight, self.bias)


# ----------------------------------------------------------
# QUANTIZED EMBEDDING LAYER — INT8
# Mirrors NVFP4's "FP8 scaling" for the embedding table.
# Embeddings are ~23M params in DistilBERT (~92 MB at FP32).
# INT8 compresses this to ~23 MB — the biggest single saving.
# ----------------------------------------------------------
class QuantizedEmbedding(nn.Module):
    """
    INT8 quantized embedding table.
    Per-row scale factors (one scale per vocabulary token).
    This mirrors the per-block scaling philosophy of NVFP4.
    """
    def __init__(self, embedding_layer: nn.Embedding):
        super().__init__()
        weight = embedding_layer.weight.data  # [vocab_size, hidden_dim]

        q_min, q_max = -128, 127  # INT8 symmetric range

        # Per-row (per-token) absolute max for scale
        abs_max = weight.abs().max(dim=1, keepdim=True).values  # [vocab, 1]
        scale   = abs_max / 127.0                               # [vocab, 1]
        scale   = scale.clamp(min=1e-8)

        q_weight = torch.clamp(
            torch.round(weight / scale), q_min, q_max
        ).to(torch.int8)

        self.register_buffer("q_weight",       q_weight)
        self.register_buffer("scale",          scale)
        self.num_embeddings  = embedding_layer.num_embeddings
        self.embedding_dim   = embedding_layer.embedding_dim
        self.padding_idx     = embedding_layer.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Lookup quantized rows, dequantize with per-row scale
        q_rows = self.q_weight[input_ids]               # [..., hidden]
        scales = self.scale[input_ids]                  # [..., 1]
        return q_rows.float() * scales


# ----------------------------------------------------------
# HELPER: Replace all Linear layers with quantized versions
# ----------------------------------------------------------
def replace_linear_layers(model: nn.Module, config: dict) -> nn.Module:
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        bits = config.get(name, 8)  # Default to 8 if not in config
        parent = model
        parts  = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], QuantizedLinear(module, bits))
        print(f"  [LINEAR ] {name:<60} → {bits}-bit")
        replaced += 1
    print(f"\n  Total linear layers replaced: {replaced}")
    return model


# ----------------------------------------------------------
# HELPER: Replace all Embedding layers with INT8 versions
# ----------------------------------------------------------
def replace_embedding_layers(model: nn.Module) -> nn.Module:
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Embedding):
            continue
        parent = model
        parts  = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], QuantizedEmbedding(module))
        vocab, dim = module.num_embeddings, module.embedding_dim
        print(f"  [EMBED  ] {name:<60} → INT8  ({vocab}×{dim})")
        replaced += 1
    print(f"\n  Total embedding layers replaced: {replaced}")
    return model


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load fine-tuned baseline
    print("=" * 60)
    print("Loading baseline model...")
    print("=" * 60)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    state = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded: {BASELINE_PATH}\n")

    # 2. Replace embedding layers → INT8
    print("=" * 60)
    print("STEP 1 — Quantizing Embeddings to INT8")
    print("  (mirrors NVFP4 FP8 scaling for non-critical large tables)")
    print("=" * 60)
    model = replace_embedding_layers(model)

    # 3. Replace linear layers → SA mixed precision
    print("\n" + "=" * 60)
    print("STEP 2 — Applying SA Mixed Precision to Transformer Linears")
    print("  (mirrors NVFP4 micro-block precision assignment)")
    print("=" * 60)
    model = replace_linear_layers(model, SA_CONFIG)

    # 4. Compute and display effective size
    print("\n" + "=" * 60)
    print("STEP 3 — Computing Effective Model Size")
    print("=" * 60)

    total_bits   = 0
    total_params = 0
    layer_stats  = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            params = module.q_weight.numel()
            bits   = module.bits
            total_bits   += params * bits
            total_params += params
            layer_stats[name] = {"type": "linear", "bits": bits, "params": params}

        elif isinstance(module, QuantizedEmbedding):
            params = module.q_weight.numel()
            total_bits   += params * 8   # INT8
            total_params += params
            layer_stats[name] = {"type": "embedding", "bits": 8, "params": params}

    # Remaining FP32 params (LayerNorm, biases, etc.)
    fp32_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "q_weight" not in name
    )
    total_bits   += fp32_params * 32
    total_params += fp32_params

    effective_bits = total_bits / total_params
    effective_size = total_bits / 8 / 1024 / 1024  # MB
    fp32_size      = total_params * 32 / 8 / 1024 / 1024

    print(f"  FP32 baseline size    : {fp32_size:.2f} MB")
    print(f"  Effective bit-width   : {effective_bits:.2f} bits/param")
    print(f"  Effective model size  : {effective_size:.2f} MB")
    print(f"  Compression ratio     : {fp32_size / effective_size:.2f}×")

    # 5. Export to ONNX
    print("\n" + "=" * 60)
    print("STEP 4 — Exporting to ONNX")
    print("=" * 60)

    dummy_ids   = torch.randint(0, 1000, (1, SEQ_LEN), dtype=torch.long).to(DEVICE)
    dummy_mask  = torch.ones((1, SEQ_LEN), dtype=torch.long).to(DEVICE)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        ONNX_OUT,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
    )
    onnx_size = os.path.getsize(ONNX_OUT) / 1024 / 1024
    print(f"  Saved : {ONNX_OUT}")
    print(f"  ONNX file size : {onnx_size:.2f} MB")

    # 6. Save config summary
    summary = {
        "strategy": {
            "embeddings":    "INT8 (per-row scale)",
            "locked_layers": "8-bit (FAR sensitivity)",
            "sa_layers":     "mixed {4,6,8} bit",
            "layernorm":     "FP32",
            "softmax":       "FP32",
        },
        "sizes": {
            "fp32_mb":         round(fp32_size, 2),
            "effective_mb":    round(effective_size, 2),
            "onnx_mb":         round(onnx_size, 2),
            "compression_ratio": round(fp32_size / effective_size, 2),
            "effective_bits":  round(effective_bits, 2),
        },
        "locked_layers":  list(LOCKED_LAYERS),
        "sa_config":      SA_CONFIG,
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Config summary saved: {SUMMARY_OUT}")

    print("\n" + "=" * 60)
    print("HYBRID EXPORT COMPLETE")
    print(f"  Model  : {ONNX_OUT}")
    print(f"  Size   : {effective_size:.2f} MB  (was {fp32_size:.2f} MB)")
    print(f"  Ratio  : {fp32_size / effective_size:.2f}× compression")
    print("=" * 60)


if __name__ == "__main__":
    main()
