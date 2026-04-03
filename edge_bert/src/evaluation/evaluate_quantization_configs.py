import torch
import torch.nn as nn
from pathlib import Path
import sys
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import json
from shared.model_workflows import load_classifier_checkpoint

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.experiment_settings import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    baseline_checkpoint_path,
    load_sa_best_config,
)

# ==========================================================
# 1. QUANTIZATION SIMULATION CLASSES
# ==========================================================
class QuantizedLinear(nn.Module):
    """
    Simulates mixed-precision quantization by clamping weights 
    to exact mathematical ranges of 4, 6, or 8-bit integers.
    """
    def __init__(self, linear_layer: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear_layer.weight.data
        
        q_min = 0
        q_max = 2 ** bits - 1
        w_min = weight.min()
        w_max = weight.max()
        
        scale = (w_max - w_min) / (q_max - q_min + 1e-8)
        zero_point = torch.round(q_min - w_min / (scale + 1e-8))
        
        q_weight = torch.clamp(
            torch.round(weight / scale + zero_point), q_min, q_max
        ).to(torch.int32)
        
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization for the forward pass
        weight = self.scale * (self.q_weight.float() - self.zero_point)
        return nn.functional.linear(x, weight, self.bias)

def apply_mixed_precision(model: nn.Module, config_dict: dict) -> nn.Module:
    """
    Iterates through the model and replaces standard Linear layers 
    with QuantizedLinear layers based on the precision map.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Default to 8-bit if the layer is somehow missing from the config
            bits = config_dict.get(name, 8) 
            
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], QuantizedLinear(module, bits))
            replaced += 1
            
    print(f"Applied mixed-precision to {replaced} linear layers.")
    return model

# ==========================================================
# 2. EVALUATION LOOP
# ==========================================================
def evaluate_model(model: nn.Module, dataloader, device="cpu"):
    """Runs the dataset through the model and calculates Accuracy & F1"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

# ==========================================================
# 3. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = DEFAULT_MODEL_NAME
    SRC_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = SRC_ROOT / "models"
    BASELINE_PATH = baseline_checkpoint_path()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running evaluation on: {DEVICE.upper()}")
    
    # --- Load GLUE SST-2 Dataset ---
    print("\nLoading GLUE SST-2 Validation Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("glue", "sst2", split="validation")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=32)
    print(f"Dataset loaded successfully: {len(dataset)} validation samples.\n")

    # --- A. Evaluate Baseline Reference ---
    print("--- 1. Evaluating Baseline Reference Checkpoint ---")
    model_base = load_classifier_checkpoint(BASELINE_PATH, device=DEVICE)
    base_acc, base_f1 = evaluate_model(model_base, dataloader, DEVICE)
    print(f"Baseline Reference -> Accuracy: {base_acc:.4f} | F1: {base_f1:.4f}\n")

    # --- B. Evaluate Greedy Model ---
    print("--- 2. Evaluating Greedy Quantized Model ---")
    with open(MODELS_DIR / "greedy_config.json", "r") as f:
        greedy_config = json.load(f)
        
    model_greedy = load_classifier_checkpoint(BASELINE_PATH, device=DEVICE)
    model_greedy = apply_mixed_precision(model_greedy, greedy_config)
    greedy_acc, greedy_f1 = evaluate_model(model_greedy, dataloader, DEVICE)
    print(f"Greedy -> Accuracy: {greedy_acc:.4f} | F1: {greedy_f1:.4f}\n")

    # --- C. Evaluate SA-Hybrid Model ---
    print("--- 3. Evaluating SA-Hybrid Quantized Model ---")
    sa_config = load_sa_best_config()

    model_sa = load_classifier_checkpoint(BASELINE_PATH, device=DEVICE)
    model_sa = apply_mixed_precision(model_sa, sa_config)
    sa_acc, sa_f1 = evaluate_model(model_sa, dataloader, DEVICE)
    print(f"SA-Hybrid -> Accuracy: {sa_acc:.4f} | F1: {sa_f1:.4f}\n")
    
    print("==================================================")
    print("FINAL EVALUATION COMPLETE")
    print("==================================================")
