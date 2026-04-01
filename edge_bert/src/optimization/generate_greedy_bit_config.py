import json
from pathlib import Path

# -----------------------------
# Sensitivity scores (paste yours)
# -----------------------------
sensitivity = {
    "distilbert.transformer.layer.0.attention.q_lin": -0.0034,
    "distilbert.transformer.layer.0.attention.k_lin": -0.0023,
    "distilbert.transformer.layer.0.attention.v_lin": -0.0011,
    "distilbert.transformer.layer.0.attention.out_lin": -0.0023,
    "distilbert.transformer.layer.0.ffn.lin1": -0.0069,
    "distilbert.transformer.layer.0.ffn.lin2": 0.0069,

    "distilbert.transformer.layer.1.attention.q_lin": -0.0046,
    "distilbert.transformer.layer.1.attention.k_lin": 0.0011,
    "distilbert.transformer.layer.1.attention.v_lin": 0.0000,
    "distilbert.transformer.layer.1.attention.out_lin": -0.0034,
    "distilbert.transformer.layer.1.ffn.lin1": 0.0138,
    "distilbert.transformer.layer.1.ffn.lin2": 0.0883,

    "distilbert.transformer.layer.2.attention.q_lin": -0.0034,
    "distilbert.transformer.layer.2.attention.k_lin": -0.0023,
    "distilbert.transformer.layer.2.attention.v_lin": -0.0069,
    "distilbert.transformer.layer.2.attention.out_lin": -0.0034,
    "distilbert.transformer.layer.2.ffn.lin1": 0.0195,
    "distilbert.transformer.layer.2.ffn.lin2": 0.0241,

    "distilbert.transformer.layer.3.attention.q_lin": -0.0011,
    "distilbert.transformer.layer.3.attention.k_lin": -0.0046,
    "distilbert.transformer.layer.3.attention.v_lin": -0.0023,
    "distilbert.transformer.layer.3.attention.out_lin": 0.0023,
    "distilbert.transformer.layer.3.ffn.lin1": 0.0023,
    "distilbert.transformer.layer.3.ffn.lin2": 0.0069,

    "distilbert.transformer.layer.4.attention.q_lin": -0.0034,
    "distilbert.transformer.layer.4.attention.k_lin": 0.0000,
    "distilbert.transformer.layer.4.attention.v_lin": 0.0000,
    "distilbert.transformer.layer.4.attention.out_lin": -0.0011,
    "distilbert.transformer.layer.4.ffn.lin1": -0.0011,
    "distilbert.transformer.layer.4.ffn.lin2": 0.0023,

    "distilbert.transformer.layer.5.attention.q_lin": 0.0023,
    "distilbert.transformer.layer.5.attention.k_lin": -0.0023,
    "distilbert.transformer.layer.5.attention.v_lin": -0.0034,
    "distilbert.transformer.layer.5.attention.out_lin": -0.0023,
    "distilbert.transformer.layer.5.ffn.lin1": -0.0034,
    "distilbert.transformer.layer.5.ffn.lin2": 0.0000,

    "pre_classifier": 0.0000,
    "classifier": -0.0011
}

# -----------------------------
# Locked sensitive layers
# -----------------------------
LOCKED = {
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin1",
}

# Remove locked layers from sorting
search_layers = {k: v for k, v in sensitivity.items() if k not in LOCKED}

# Sort by sensitivity (descending)
sorted_layers = sorted(search_layers.items(), key=lambda x: x[1], reverse=True)

n = len(sorted_layers)

top = sorted_layers[: n//3]
mid = sorted_layers[n//3 : 2*n//3]
low = sorted_layers[2*n//3 :]

greedy_config = {}

for layer,_ in top:
    greedy_config[layer] = 8

for layer,_ in mid:
    greedy_config[layer] = 6

for layer,_ in low:
    greedy_config[layer] = 4

# Add locked layers
for layer in LOCKED:
    greedy_config[layer] = 8

print("\nGreedy Bit Allocation\n")

for k,v in greedy_config.items():
    print(k,"→",v)

# Save config
with open(MODELS_DIR / "greedy_config.json","w") as f:
    json.dump(greedy_config,f,indent=4)

print(f"\nSaved to {MODELS_DIR / 'greedy_config.json'}")
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
