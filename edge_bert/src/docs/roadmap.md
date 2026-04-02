# Project Roadmap: How To Run `edge_bert`

This document is the operational runbook for the current repository layout. It explains:

- what this project does
- what must exist before each stage runs
- the exact script order
- what each script produces
- which steps are optional

All paths below are relative to the repository root:

`edge_bert/`

Most scripts assume they are launched from the repository root or from `src/`-aware paths. The safest approach is:

```powershell
cd c:\Users\skvar\Desktop\workspace\Research\edge_bert
```

## 1. What The Project Currently Supports

This repo builds and evaluates several DistilBERT variants for GLUE SST-2 style binary classification:

- FP32 baseline checkpoint
- FP32 baseline exported to ONNX
- dynamic INT8 ONNX baseline
- frozen-layer FP32 checkpoint
- frozen-layer INT8 ONNX
- greedy mixed-precision model
- simulated annealing mixed-precision model
- hybrid model with INT8 embeddings plus SA-selected linear layer precision

The repo now also supports:

- sensitivity-driven locked layers from generated analysis output
- constrained Pareto simulated annealing search
- dynamic accuracy thresholds derived from the measured baseline accuracy

## 2. Before You Run Anything

### 2.1 Python environment

There is no checked-in `requirements.txt` or `pyproject.toml` right now, so the environment must be created from the imports used in the code.

Minimum practical dependency set:

```powershell
pip install torch transformers datasets scikit-learn onnxruntime onnx matplotlib numpy tqdm codecarbon psutil
```

If you use GPU PyTorch, install the correct PyTorch build first from the official channel, then install the rest.

### 2.2 Internet requirement

The first run may download:

- `distilbert-base-uncased`
- the tokenizer for that model
- the dataset configured in `src/config/experiment_settings.json`

By default that dataset is:

- dataset name: `glue`
- dataset config: `sst2`
- evaluation split: `validation`

### 2.3 Important directories

Generated files are expected in:

- `src/models/`
- `src/results/`
- `src/figures/`

Several scripts fail if upstream artifacts are missing, so follow the execution order below.

## 3. Global Configuration

The central config file is:

[experiment_settings.json](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\config\experiment_settings.json)

This file controls:

- model name
- label count
- dataset name and split
- tokenizer max length
- quantization bit options
- sensitivity lock threshold
- simulated annealing settings
- dynamic threshold drop percentages
- evaluation batch size and latency settings

The Python loader for those values is:

[experiment_settings.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\shared\experiment_settings.py)

### 3.1 If you want another dataset

Before running the pipeline, update:

- `model.name`
- `model.num_labels`
- `dataset.name`
- `dataset.config`
- `dataset.evaluation_split`
- `dataset.max_length`

Be aware that many scripts are still DistilBERT-specific in structure, even though configuration is now centralized. The current repo is most reliable with `distilbert-base-uncased` on SST-2.

## 4. Recommended Execution Paths

There are three practical ways to run the project.

### Path A: Reproduce from scratch

Use this if you want to regenerate checkpoints, search outputs, ONNX exports, metrics, and figures.

### Path B: Rebuild downstream artifacts from existing checkpoints

Use this if `src/models/baseline_best.pt` and `src/models/frozen_best.pt` already exist.

### Path C: Only evaluate already-exported models

Use this if ONNX files already exist in `src/models/`.

The full detailed flow below is written for Path A. You can skip upstream stages if the required files already exist.

## 5. Stage-by-Stage Run Order

## 5.1 Stage 0: Sanity-check existing assets

Look in:

- `src/models/`
- `src/results/`
- `src/figures/`

Typical currently expected model artifacts:

- `baseline_best.pt`
- `baseline_fp32.onnx`
- `baseline_int8.onnx`
- `frozen_best.pt`
- `frozen_fp32.onnx`
- `frozen_int8.onnx`
- `greedy_config.json`
- `greedy_quant_model.pt`
- `greedy_quant_model.onnx`
- `mixed_precision_real_quant.onnx`
- `hybrid_quant_model.onnx`

If the checkpoints are missing, start from Stage 1.

## 5.2 Stage 1: Train the baseline model

Script:

[train_baseline_model.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\training\train_baseline_model.py)

Command:

```powershell
python src/training/train_baseline_model.py
```

What it does:

- loads the configured dataset
- tokenizes text with the configured tokenizer
- trains `DistilBertForSequenceClassification`
- tracks validation accuracy and F1
- saves the best checkpoint
- tracks emissions through `codecarbon`

Output:

- `src/models/baseline_best.pt`

Notes:

- the script uses `batch_size=16`
- it trains for `3` epochs
- it applies early stopping with patience `2`
- if CUDA is available, it trains on GPU

## 5.3 Stage 2: Evaluate the baseline checkpoint

Script:

[evaluate_baseline_checkpoint.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\evaluate_baseline_checkpoint.py)

Command:

```powershell
python src/evaluation/evaluate_baseline_checkpoint.py
```

Dependency:

- `src/models/baseline_best.pt`

What it reports:

- validation accuracy
- validation F1
- checkpoint file size
- average CPU latency

This script does not save a JSON result file by itself. It is mainly a checkpoint sanity check.

## 5.4 Stage 3: Train the frozen model

Script:

[train_frozen_model.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\training\train_frozen_model.py)

Command:

```powershell
python src/training/train_frozen_model.py
```

What it does:

- trains another DistilBERT classifier
- freezes embeddings
- freezes the first 2 transformer layers
- saves the best frozen checkpoint

Output:

- `src/models/frozen_best.pt`

## 5.5 Stage 4: Evaluate the frozen checkpoint

Script:

[evaluate_frozen_checkpoint.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\evaluate_frozen_checkpoint.py)

Command:

```powershell
python src/evaluation/evaluate_frozen_checkpoint.py
```

Dependency:

- `src/models/frozen_best.pt`

Purpose:

- confirms the frozen checkpoint is valid before ONNX export

## 5.6 Stage 5: Export baseline and frozen checkpoints to ONNX

Scripts:

- [export_baseline_checkpoint_to_onnx.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_baseline_checkpoint_to_onnx.py)
- [export_frozen_checkpoint_to_onnx.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_frozen_checkpoint_to_onnx.py)

Commands:

```powershell
python src/export/export_baseline_checkpoint_to_onnx.py
python src/export/export_frozen_checkpoint_to_onnx.py
```

Outputs:

- `src/models/baseline_fp32.onnx`
- `src/models/frozen_fp32.onnx`

## 5.7 Stage 6: Quantize baseline and frozen ONNX models to dynamic INT8

Scripts:

- [quantize_baseline_onnx_to_int8.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\quantize_baseline_onnx_to_int8.py)
- [quantize_frozen_onnx_to_int8.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\quantize_frozen_onnx_to_int8.py)

Commands:

```powershell
python src/export/quantize_baseline_onnx_to_int8.py
python src/export/quantize_frozen_onnx_to_int8.py
```

Dependencies:

- `src/models/baseline_fp32.onnx`
- `src/models/frozen_fp32.onnx`

Outputs:

- `src/models/baseline_int8.onnx`
- `src/models/frozen_int8.onnx`

## 5.8 Stage 7: Run layer sensitivity analysis

Script:

[analyze_layer_sensitivity.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\analysis\analyze_layer_sensitivity.py)

Command:

```powershell
python src/analysis/analyze_layer_sensitivity.py
```

Dependency:

- `src/models/baseline_best.pt`

What it does:

- measures baseline validation accuracy
- fake-quantizes one linear layer at a time
- measures accuracy drop for each layer
- ranks layers by sensitivity
- derives locked layers using the configured `lock_threshold`

Output:

- `src/results/sensitivity_results.json`

Why this matters:

- greedy bit assignment uses this file
- SA exports use the locked-layer set from this file

## 5.9 Stage 8: Build the greedy mixed-precision config

Script:

[generate_greedy_bit_config.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\optimization\generate_greedy_bit_config.py)

Command:

```powershell
python src/optimization/generate_greedy_bit_config.py
```

Dependency:

- `src/results/sensitivity_results.json`

What it does:

- loads per-layer sensitivity
- excludes locked layers from ranking
- sorts remaining layers by sensitivity
- applies the configured greedy allocation strategy
- currently supports `tertiles`

Output:

- `src/models/greedy_config.json`

## 5.10 Stage 9: Optional sanity check of the old scalar energy function

Script:

[sanity_check_energy_function.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\optimization\sanity_check_energy_function.py)

Command:

```powershell
python src/optimization/sanity_check_energy_function.py
```

This is optional now. It is mainly helpful if you want to inspect:

- all-8-bit simulated candidate behavior
- all-4-bit simulated candidate behavior
- one random candidate behavior

This is not the main search path anymore.

## 5.11 Stage 10: Run constrained Pareto simulated annealing

Script:

[run_simulated_annealing_search.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\optimization\run_simulated_annealing_search.py)

Command:

```powershell
python src/optimization/run_simulated_annealing_search.py
```

Dependencies:

- `src/models/baseline_best.pt`
- `src/results/sensitivity_results.json` if you want generated locked layers

What it now does:

- measures the baseline model accuracy on the configured validation set
- generates threshold values dynamically from baseline accuracy using:
  - `threshold_drop_percentages` in config
- estimates lower and upper bounds for size and latency
- runs constrained Pareto SA separately for each threshold
- keeps only feasible candidates where:
  - `candidate_accuracy >= threshold_accuracy`
- records a Pareto frontier per threshold
- selects one default representative result for downstream scripts

Current threshold formula:

```text
threshold_accuracy = baseline_accuracy - (drop_percent / 100)
```

Key outputs:

- `src/results/sa_search_results.json`
- `src/results/sa_best_config.json`
- `src/figures/sa_search_convergence.png`
- `src/figures/sa_pareto_frontiers.png`

Important detail:

- `sa_best_config.json` is the compatibility bridge for downstream scripts
- it stores one default representative mixed-precision config
- the full multi-threshold result set lives in `sa_search_results.json`

## 5.12 Stage 11: Export the greedy mixed-precision model

This is a two-step process.

### Step 11A: Create quantized checkpoint

Script:

[export_greedy_quantized_checkpoint.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_greedy_quantized_checkpoint.py)

Command:

```powershell
python src/export/export_greedy_quantized_checkpoint.py
```

Dependencies:

- `src/models/baseline_best.pt`
- `src/models/greedy_config.json`

Output:

- `src/models/greedy_quant_model.pt`

### Step 11B: Export greedy checkpoint to ONNX

Script:

[export_greedy_checkpoint_to_onnx.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_greedy_checkpoint_to_onnx.py)

Command:

```powershell
python src/export/export_greedy_checkpoint_to_onnx.py
```

Dependency:

- `src/models/greedy_quant_model.pt`

Output:

- `src/models/greedy_quant_model.onnx`

## 5.13 Stage 12: Export the SA mixed-precision model

Script:

[export_sa_quantized_onnx.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_sa_quantized_onnx.py)

Command:

```powershell
python src/export/export_sa_quantized_onnx.py
```

Dependencies:

- `src/models/baseline_best.pt`
- `src/results/sa_best_config.json`

What it does:

- loads the baseline checkpoint
- loads the default SA config
- forces locked layers to 8-bit
- replaces linear layers with quantized modules
- exports the ONNX graph

Output:

- `src/models/mixed_precision_real_quant.onnx`

## 5.14 Stage 13: Export the hybrid ONNX model

Script:

[export_hybrid_quantized_onnx.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\export\export_hybrid_quantized_onnx.py)

Command:

```powershell
python src/export/export_hybrid_quantized_onnx.py
```

Dependencies:

- `src/models/baseline_best.pt`
- `src/results/sa_best_config.json`

What it does:

- quantizes embeddings to INT8
- uses the SA-selected mixed precision for linear layers
- keeps locked layers at 8-bit
- saves a summary JSON alongside the ONNX model

Outputs:

- `src/models/hybrid_quant_model.onnx`
- `src/models/hybrid_config_summary.json`

## 5.15 Stage 14: Evaluate all ONNX models together

Script:

[evaluate_all_onnx_models.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\evaluate_all_onnx_models.py)

Command:

```powershell
python src/evaluation/evaluate_all_onnx_models.py
```

What it expects:

- `baseline_fp32.onnx`
- `baseline_int8.onnx`
- `frozen_fp32.onnx`
- `frozen_int8.onnx`
- `greedy_quant_model.onnx`
- `mixed_precision_real_quant.onnx`
- `hybrid_quant_model.onnx`

What it computes:

- accuracy
- F1
- ROC-AUC
- average latency
- p50 latency
- p95 latency
- ONNX file size

Outputs:

- `src/results/all_model_results.json`
- `src/results/all_model_results.txt`

If some ONNX files are missing:

- the script does not fully fail
- it marks those entries as missing and continues

## 5.16 Stage 15: Compare theoretical model sizes

Script:

[compare_model_sizes.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\analysis\compare_model_sizes.py)

Command:

```powershell
python src/analysis/compare_model_sizes.py
```

Dependencies:

- `src/models/greedy_config.json`
- `src/results/sa_best_config.json`
- ONNX files if you want actual ONNX sizes included

What it computes:

- theoretical effective size in MB
- effective bits
- compression ratio
- ONNX file size when the file exists

Outputs:

- `src/results/size_comparison.json`
- `src/results/size_comparison.txt`

## 5.17 Stage 16: Generate final paper/report figures

Script:

[generate_paper_figures.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\analysis\generate_paper_figures.py)

Command:

```powershell
python src/analysis/generate_paper_figures.py
```

Dependencies:

- `src/results/all_model_results.json`
- `src/results/size_comparison.json`
- `src/results/sa_best_config.json`
- `src/results/sensitivity_results.json` if you want accurate locked-layer annotations

Outputs:

- `src/figures/fig1_tradeoff_curve.png`
- `src/figures/fig2_bit_heatmap.png`
- `src/figures/fig3_compression_bars.png`
- `src/figures/fig4_latency_bars.png`
- `src/figures/fig5_accuracy_drop.png`

## 6. Fast Reproduction Order

If checkpoints do not exist, use this exact order:

```powershell
python src/training/train_baseline_model.py
python src/evaluation/evaluate_baseline_checkpoint.py
python src/training/train_frozen_model.py
python src/evaluation/evaluate_frozen_checkpoint.py
python src/export/export_baseline_checkpoint_to_onnx.py
python src/export/quantize_baseline_onnx_to_int8.py
python src/export/export_frozen_checkpoint_to_onnx.py
python src/export/quantize_frozen_onnx_to_int8.py
python src/analysis/analyze_layer_sensitivity.py
python src/optimization/generate_greedy_bit_config.py
python src/optimization/run_simulated_annealing_search.py
python src/export/export_greedy_quantized_checkpoint.py
python src/export/export_greedy_checkpoint_to_onnx.py
python src/export/export_sa_quantized_onnx.py
python src/export/export_hybrid_quantized_onnx.py
python src/evaluation/evaluate_all_onnx_models.py
python src/analysis/compare_model_sizes.py
python src/analysis/generate_paper_figures.py
```

## 7. Minimal Evaluation-Only Order

If the ONNX files already exist and you only want metrics plus figures:

```powershell
python src/evaluation/evaluate_all_onnx_models.py
python src/analysis/compare_model_sizes.py
python src/analysis/generate_paper_figures.py
```

## 8. Outputs You Should Expect At The End

### In `src/models/`

- PyTorch checkpoints
- ONNX exports
- greedy config
- hybrid summary JSON

### In `src/results/`

- `sensitivity_results.json`
- `sa_search_results.json`
- `sa_best_config.json`
- `all_model_results.json`
- `size_comparison.json`

### In `src/figures/`

- SA convergence plot
- constrained Pareto frontier plot
- five comparison/report figures

## 9. Common Failure Points

### Missing checkpoint

Symptom:

- export or evaluation scripts fail with file-not-found errors for `baseline_best.pt` or `frozen_best.pt`

Fix:

- rerun the corresponding training stage

### Missing sensitivity results

Symptom:

- greedy config generation fails
- locked layers fall back to config defaults instead of measured sensitivity

Fix:

- rerun `python src/analysis/analyze_layer_sensitivity.py`

### Missing SA config

Symptom:

- SA export scripts fail because `sa_best_config.json` does not exist

Fix:

- rerun `python src/optimization/run_simulated_annealing_search.py`

### Missing ONNX files

Symptom:

- `evaluate_all_onnx_models.py` marks models as missing

Fix:

- rerun the relevant export stage

### Wrong dataset/model configuration

Symptom:

- label mismatch
- tensor shape mismatch
- unstable comparison across scripts

Fix:

- update `src/config/experiment_settings.json`
- then regenerate artifacts from the earliest affected stage

## 10. Practical Notes

- The SA search is the slowest stage because candidate evaluation still uses validation-set model passes.
- The project is now more dataset-aware than before, but not every script is fully architecture-agnostic.
- If you change model family or number of labels, regenerate everything from Stage 1 or Stage 7 depending on the scope of the change.
- `src/llama.cpp/` is not part of the main DistilBERT quantization pipeline documented here.

## 11. Best Current Default Workflow

If your goal is the full current intended pipeline, this is the cleanest high-level sequence:

1. Train baseline and frozen checkpoints.
2. Export FP32 and INT8 baseline/frozen ONNX models.
3. Run sensitivity analysis.
4. Generate greedy config.
5. Run constrained Pareto SA search with dynamic thresholds.
6. Export greedy, SA, and hybrid mixed-precision ONNX models.
7. Evaluate all ONNX models together.
8. Compute size comparisons.
9. Generate figures.

That gives you the complete artifact chain needed for experiments, comparisons, and paper-style plots.
