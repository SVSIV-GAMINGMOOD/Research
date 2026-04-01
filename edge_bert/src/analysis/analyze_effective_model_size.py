import torch
from transformers import DistilBertForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

# ==========================================================
# SA CONFIGURATION (from your script)
# ==========================================================

sa_config = {
    "distilbert.transformer.layer.0.attention.q_lin": 8,
    "distilbert.transformer.layer.0.attention.k_lin": 6,
    "distilbert.transformer.layer.0.attention.v_lin": 4,
    "distilbert.transformer.layer.0.attention.out_lin": 8,
    "distilbert.transformer.layer.0.ffn.lin1": 4,
    "distilbert.transformer.layer.0.ffn.lin2": 6,

    "distilbert.transformer.layer.1.attention.q_lin": 6,
    "distilbert.transformer.layer.1.attention.k_lin": 4,
    "distilbert.transformer.layer.1.attention.v_lin": 4,
    "distilbert.transformer.layer.1.attention.out_lin": 6,

    "distilbert.transformer.layer.2.attention.q_lin": 4,
    "distilbert.transformer.layer.2.attention.k_lin": 6,
    "distilbert.transformer.layer.2.attention.v_lin": 4,
    "distilbert.transformer.layer.2.attention.out_lin": 6,

    "distilbert.transformer.layer.3.attention.q_lin": 8,
    "distilbert.transformer.layer.3.attention.k_lin": 6,
    "distilbert.transformer.layer.3.attention.v_lin": 4,
    "distilbert.transformer.layer.3.attention.out_lin": 8,
    "distilbert.transformer.layer.3.ffn.lin1": 8,
    "distilbert.transformer.layer.3.ffn.lin2": 6,

    "distilbert.transformer.layer.4.attention.q_lin": 8,
    "distilbert.transformer.layer.4.attention.k_lin": 8,
    "distilbert.transformer.layer.4.attention.v_lin": 8,
    "distilbert.transformer.layer.4.attention.out_lin": 4,
    "distilbert.transformer.layer.4.ffn.lin1": 4,
    "distilbert.transformer.layer.4.ffn.lin2": 6,

    "distilbert.transformer.layer.5.attention.q_lin": 4,
    "distilbert.transformer.layer.5.attention.k_lin": 4,
    "distilbert.transformer.layer.5.attention.v_lin": 8,
    "distilbert.transformer.layer.5.attention.out_lin": 6,
    "distilbert.transformer.layer.5.ffn.lin1": 6,
    "distilbert.transformer.layer.5.ffn.lin2": 4,

    "pre_classifier": 4,
    "classifier": 8,
}

LOCKED_LAYERS = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1",
}

# ==========================================================
# GREEDY CONFIG (your result)
# ==========================================================

greedy_config = {
    "distilbert.transformer.layer.2.ffn.lin2": 8,
    "distilbert.transformer.layer.1.ffn.lin1": 8,
    "distilbert.transformer.layer.2.ffn.lin1": 8,
    "distilbert.transformer.layer.1.ffn.lin2": 8,
}

DEFAULT_BITS = 32


# ==========================================================
# SIZE COMPUTATION
# ==========================================================

def compute_size(model, config=None):

    total_bits = 0
    total_params = 0

    for name, param in model.named_parameters():

        params = param.numel()
        total_params += params

        layer_name = name.replace(".weight", "").replace(".bias", "")

        if config and layer_name in config:
            bits = config[layer_name]
        else:
            bits = DEFAULT_BITS

        total_bits += params * bits

    size_mb = total_bits / 8 / 1024 / 1024
    return size_mb, total_params


# ==========================================================
# LOAD MODEL
# ==========================================================

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("\nModel Loaded\n")

# ==========================================================
# BASELINES
# ==========================================================

total_params = sum(p.numel() for p in model.parameters())

fp32_size = total_params * 32 / 8 / 1024 / 1024
int8_size = total_params * 8 / 8 / 1024 / 1024

# ==========================================================
# GREEDY
# ==========================================================

greedy_size, _ = compute_size(model, greedy_config)

# ==========================================================
# SA
# ==========================================================

# enforce locked layers
for layer in LOCKED_LAYERS:
    sa_config[layer] = 8

sa_size, _ = compute_size(model, sa_config)

# ==========================================================
# PRINT RESULTS
# ==========================================================

print("===== EFFECTIVE MODEL SIZE =====\n")

print(f"FP32 Model        : {fp32_size:.2f} MB")
print(f"INT8 Model        : {int8_size:.2f} MB")
print(f"Greedy Mixed      : {greedy_size:.2f} MB")
print(f"SA Mixed          : {sa_size:.2f} MB")

print("\nTotal Parameters:", total_params)