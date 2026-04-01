import json
from pathlib import Path

BIT_MAP = {
    8: "Q8_0",
    6: "Q6_K",
    4: "Q4_K"
}

def translate(name):
    # Only process transformer layers
    if "distilbert.transformer.layer" in name:
        name = name.replace("distilbert.transformer.layer.", "blk.")
        name = name.replace("attention.q_lin", "attn_q")
        name = name.replace("attention.k_lin", "attn_k")
        name = name.replace("attention.v_lin", "attn_v")
        name = name.replace("attention.out_lin", "attn_output")
        name = name.replace("ffn.lin1", "ffn_up")
        name = name.replace("ffn.lin2", "ffn_down")
        
        return name + ".weight"

    # We deleted the classifier heads from the base model earlier!
    # Returning None ensures they don't get written to the map file.
    if name == "pre_classifier" or name == "classifier":
        return None

    return None


def build_tensor_map(config_file, output_file, dict_key=None):
    with open(config_file, "r") as f:
        config = json.load(f)

    # If the JSON is nested (like hybrid_config_summary.json), extract the layer dict
    if dict_key and dict_key in config:
        layer_dict = config[dict_key]
    else:
        # Fallback if it's already a flat dictionary (like greedy might be)
        layer_dict = config

    with open(output_file, "w") as f:
        # 1. FIXED: Use equals sign for embeddings
        f.write("token_embd.weight=Q8_0\n")

        for layer, bits in layer_dict.items():
            gguf_name = translate(layer)

            if gguf_name is None:
                continue

            qtype = BIT_MAP.get(bits, "Q8_0")

            # 2. FIXED: Use equals sign for all other layers
            f.write(f"{gguf_name}={qtype}\n")


if __name__ == "__main__":
    src_root = Path(__file__).resolve().parents[1]
    # Pass "sa_config" so it extracts the correct dictionary from your nested JSON
    build_tensor_map(
        src_root / "models" / "hybrid_config_summary.json",
        src_root / "artifacts" / "tensor_types_hybrid.txt",
        dict_key="sa_config" 
    )

    # Assuming greedy_config is just a flat dict, we don't need dict_key.
    # If it is nested, add dict_key="your_greedy_key" here too!
    build_tensor_map(
        src_root / "models" / "greedy_config.json",
        src_root / "artifacts" / "tensor_types_greedy.txt"
    )

    print("Tensor maps created successfully!")
