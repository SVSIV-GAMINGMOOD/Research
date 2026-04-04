# Project Roadmap: Primary Metrics, Supplementary Runtime Benchmarks, and Full Result Collection

This document describes the current intended experiment design for `edge_bert`.

It now follows a two-track policy:

- `ONNX` is the primary backend for research metrics.
- `llama.cpp` / `GGUF` is a supplementary runtime benchmark backend.

It also formalizes the precision policy:

- primary deployment baseline: `FP16`
- research reference: `FP32`
- optional FP8 experiment track:
  - `Greedy Mixed`
  - `SA (v1)`
  - `Hybrid SA (ours)`

The `results/` folder is the canonical location for all experiment outputs.

## 1. What The Project Measures

The repo currently supports DistilBERT-based binary classification experiments on SST-2 with:

- baseline checkpoint training
- frozen checkpoint training
- ONNX export and INT8 export
- greedy mixed-precision quantization
- simulated annealing mixed-precision quantization
- hybrid INT8 embedding plus SA linear-layer quantization
- sensitivity analysis
- size analysis
- multi-threshold constrained Pareto search

The project now separates two questions clearly:

### Primary question

How do different quantization policies affect:

- task accuracy
- F1
- ROC-AUC
- model size
- deployment latency

This is answered with the ONNX evaluation pipeline.

### Supplementary question

How do selected mixed-precision policies behave in an alternative deployment runtime?

This is answered with `llama.cpp` / GGUF benchmarking, but only as a supplementary systems result.

## 2. Why ONNX Stays The Main Research Backend

Your current project is a sequence-classification pipeline.

That means the main benchmark must continue to evaluate:

- the actual classifier
- on the actual dataset
- with actual task metrics

The current ONNX evaluator does exactly that.

`llama.cpp` is still valuable, but mostly for:

- runtime benchmarking
- systems portability
- supplementary deployment analysis

It should not replace ONNX for the core research claim.

## 3. Precision Policy

### Primary baseline

Use `FP16` as the main deployment-oriented baseline.

This is the preferred practical baseline because it reflects modern deployment better than `FP32`.

### Reference point

Keep one `FP32` model as a research reference.

This is useful for:

- error analysis
- understanding quantization loss
- paper figures
- historical comparison

### Optional FP8 track

FP8 is not required for the current pipeline, but if you add it later, limit it to:

- `Greedy Mixed`
- `SA (v1)`
- `Hybrid SA (ours)`

Reason:

- these are the models where low-precision experiments are most relevant
- baseline and frozen variants already anchor the comparison
- the optional FP8 branch should extend the research, not replace the established baselines

## 4. Central Configuration

Main config:

[experiment_settings.json](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\config\experiment_settings.json)

Python loader:

[experiment_settings.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\shared\experiment_settings.py)

The config now covers:

- model identity
- dataset identity
- sequence length
- quantization bits
- sensitivity lock threshold
- simulated annealing settings
- dynamic accuracy-threshold generation
- evaluation settings
- experiment tracking strategy
- active training regimen selection

Downstream checkpoint consumers now use:

- `training.active_regimen`

That means exports, reference evaluation, sensitivity analysis, and related checkpoint-based scripts will read from:

- `src/models/realistic/` when `active_regimen = "realistic"`
- `src/models/controlled/` when `active_regimen = "controlled"`

This lets you keep both training regimens on disk without changing code in every downstream script.

Tracking settings now define:

- primary metrics backend
- supplementary runtime backend
- primary baseline precision
- FP32 reference precision
- optional FP8 model list
- canonical result filenames

## 5. Results Folder Policy

The `results/` folder is now intended to contain outputs from every major experimental layer.

Current key files:

- `reference_model_results.json`
- `all_model_results.json`
- `size_comparison.json`
- `experiment_registry.json`
- `gguf_runtime_results.json`
- `training_histories/realistic/*.json`
- `training_histories/controlled/*.json`

Supporting files when generated:

- `sensitivity_results.json`
- `sa_search_results.json`
- `sa_best_config.json`

See:

[README.md](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\results\README.md)

## 6. End-To-End Experiment Flow

Run from repository root:

```powershell
cd c:\Users\skvar\Desktop\workspace\Research\edge_bert
```

## 6.1 Environment setup

This repo does not yet include a formal dependency lockfile.

Minimum practical install set:

```powershell
pip install torch transformers datasets scikit-learn onnxruntime onnx matplotlib numpy tqdm codecarbon psutil
```

If you use GPU PyTorch, install the appropriate CUDA build first.

## 6.2 Stage A: Train checkpoints

### Realistic baseline checkpoint

```powershell
python src/training/realistic/train_baseline_model.py
```

Output:

- `src/models/realistic/baseline_best.pt`
- `src/results/training_histories/realistic/baseline_history.json`

Training behavior:

- `max_epochs = 20`
- early stopping with `patience = 2`
- best validation checkpoint is saved and reported

### Realistic frozen checkpoint

```powershell
python src/training/realistic/train_frozen_model.py
```

Output:

- `src/models/realistic/frozen_best.pt`
- `src/results/training_histories/realistic/frozen_history.json`

### Controlled baseline checkpoint

```powershell
python src/training/controlled/train_baseline_model.py
```

Output:

- `src/models/controlled/baseline_best.pt`
- `src/results/training_histories/controlled/baseline_history.json`

Training behavior:

- fixed `10` epochs
- no early stopping
- best validation checkpoint is still saved and reported

### Controlled frozen checkpoint

```powershell
python src/training/controlled/train_frozen_model.py
```

Output:

- `src/models/controlled/frozen_best.pt`
- `src/results/training_histories/controlled/frozen_history.json`

Each training history JSON now records:

- epoch-by-epoch validation accuracy and F1
- cumulative training steps
- best epoch and best global step
- total validation passes
- total validation examples processed

## 6.3 Stage B: Save checkpoint-level reference results

Script:

[evaluate_reference_models.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\evaluate_reference_models.py)

Command:

```powershell
python src/evaluation/evaluate_reference_models.py
```

What it records:

- `Primary Baseline (FP16)`:
  - measured when supported by runtime
  - otherwise stored as not measured with reason
- `FP32 Research Reference`
- `Frozen Checkpoint (FP32)`

Output:

- `src/results/reference_model_results.json`

Important note:

On a CPU-only environment, `FP16` is recorded as not measured. That is expected and cleaner than storing misleading CPU-half numbers.

If you want to stage the result file without running any model evaluation yet, use:

```powershell
python src/evaluation/evaluate_reference_models.py --prepare-only
```

That writes `pending_manual_run` entries so the results folder and registry stay organized while you defer the expensive runs.

## 6.4 Stage C: Export ONNX models

### Baseline FP32 ONNX

```powershell
python src/export/export_baseline_checkpoint_to_onnx.py
```

Output:

- `src/models/baseline_fp32.onnx`

### Frozen FP32 ONNX

```powershell
python src/export/export_frozen_checkpoint_to_onnx.py
```

Output:

- `src/models/frozen_fp32.onnx`

### INT8 ONNX variants

```powershell
python src/export/quantize_baseline_onnx_to_int8.py
python src/export/quantize_frozen_onnx_to_int8.py
```

Outputs:

- `src/models/baseline_int8.onnx`
- `src/models/frozen_int8.onnx`

## 6.5 Stage D: Generate sensitivity and quantization search artifacts

### Sensitivity analysis

```powershell
python src/analysis/analyze_layer_sensitivity.py
```

Output:

- `src/results/sensitivity_results.json`

### Greedy config

```powershell
python src/optimization/generate_greedy_bit_config.py
```

Output:

- `src/models/greedy_config.json`

### Constrained Pareto SA search

```powershell
python src/optimization/run_simulated_annealing_search.py
```

Outputs:

- `src/results/sa_search_results.json`
- `src/results/sa_best_config.json`
- `src/figures/sa_search_convergence.png`
- `src/figures/sa_pareto_frontiers.png`

The SA search now derives thresholds dynamically from measured baseline accuracy.

## 6.6 Stage E: Export mixed-precision models

### Greedy mixed

```powershell
python src/export/export_greedy_quantized_checkpoint.py
python src/export/export_greedy_checkpoint_to_onnx.py
```

Outputs:

- `src/models/greedy_quant_model.pt`
- `src/models/greedy_quant_model.onnx`

### SA mixed

```powershell
python src/export/export_sa_quantized_onnx.py
```

Output:

- `src/models/mixed_precision_real_quant.onnx`

### Hybrid INT8+SA

```powershell
python src/export/export_hybrid_quantized_onnx.py
```

Outputs:

- `src/models/hybrid_quant_model.onnx`
- `src/models/hybrid_config_summary.json`

## 6.7 Stage F: Run the primary ONNX experiment set

Script:

[evaluate_all_onnx_models.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\evaluate_all_onnx_models.py)

Command:

```powershell
python src/evaluation/evaluate_all_onnx_models.py
```

This is the primary research result table.

It evaluates:

- `FP32 Baseline`
- `INT8 Uniform`
- `FAR Frozen FP32`
- `FAR Frozen INT8`
- `Greedy Mixed`
- `SA (v1)`
- `Hybrid SA (ours)`

Outputs:

- `src/results/all_model_results.json`
- `src/results/all_model_results.txt`

## 6.8 Stage G: Size comparison and figure generation

### Size comparison

```powershell
python src/analysis/compare_model_sizes.py
```

Outputs:

- `src/results/size_comparison.json`
- `src/results/size_comparison.txt`

### Figures

```powershell
python src/analysis/generate_paper_figures.py
python src/analysis/plot_training_convergence.py
```

Outputs:

- `src/figures/fig1_tradeoff_curve.png`
- `src/figures/fig2_bit_heatmap.png`
- `src/figures/fig3_compression_bars.png`
- `src/figures/fig4_latency_bars.png`
- `src/figures/fig5_accuracy_drop.png`
- `src/figures/training_convergence_accuracy_vs_steps.png`

## 6.9 Stage H: Build a complete experiment registry

Script:

[build_experiment_registry.py](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\evaluation\build_experiment_registry.py)

Command:

```powershell
python src/evaluation/build_experiment_registry.py
```

This script consolidates:

- reference checkpoint results
- ONNX evaluation results
- size comparison results
- created checkpoint artifacts
- created config artifacts
- supplementary GGUF benchmark results and provenance
- optional FP8-track placeholders

Outputs:

- `src/results/experiment_registry.json`
- `src/results/gguf_runtime_results.json`

This is the file set you should use as your master experiment record.

## 7. Supplementary GGUF / llama.cpp Runtime Benchmarking

This is an additional benchmarking track, not the main scientific evaluation path.

Use it only after:

- the ONNX experiment set is complete
- the conversion path is validated
- you are sure the GGUF-exported model still represents the same quantization policy

Recommended scope:

- `Primary Baseline (FP16 GGUF)`
- `Greedy Mixed`
- `SA (v1)`
- `Hybrid SA (ours)`

Do not use GGUF runtime benchmarks as a substitute for:

- SST-2 accuracy
- F1
- ROC-AUC
- classifier-faithful evaluation

Use them for:

- runtime throughput
- alternative backend latency
- deployment discussion
- supplementary systems evidence

### Practical GGUF Command Flow

If you want to run the supplementary `llama.cpp` track in this repo, use this order:

1. Export the cleaned DistilBERT backbone in Hugging Face format:

```powershell
python src/export/export_baseline_to_huggingface.py
```

This writes the base model to `src/models/distilbert_hf_format/` and intentionally excludes the classifier-only head layers for GGUF export.

2. Generate tensor-type maps for the mixed-precision presets:

```powershell
powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 greedy
powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 sa
powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 hybrid
```

This creates:

- `src/llama.cpp/tensor_types_greedy.txt`
- `src/llama.cpp/tensor_types_sa.txt`
- `src/llama.cpp/tensor_types_hybrid.txt`

3. Convert the Hugging Face export to GGUF with the corresponding tensor map:

```powershell
powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 greedy
powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 sa
powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 hybrid
```

Default outputs:

- `src/llama.cpp/distilbert-greedy.gguf`
- `src/llama.cpp/distilbert-sa.gguf`
- `src/llama.cpp/distilbert-hybrid.gguf`

4. If you want the direct Python conversion command instead of the wrapper, use:

```powershell
python src/llama.cpp/convert_hf_to_gguf.py `
  src/models/distilbert_hf_format `
  --outfile src/llama.cpp/distilbert-hybrid.gguf `
  --outtype f16 `
  --tensor-types-file src/llama.cpp/tensor_types_hybrid.txt
```

Swap in `tensor_types_greedy.txt` or `tensor_types_sa.txt` and the corresponding output filename for the other presets.

5. Validate conversion fidelity before reporting runtime results.

At minimum, verify that:

- the GGUF file was produced successfully
- the expected tensor map was applied
- the exported model still represents the intended mixed-precision policy

6. Run runtime-only benchmarking and consolidate the results into `src/results/gguf_runtime_results.json`.

Example:

```powershell
bash src/llama.cpp/benchmark_distilbert_embeddings.sh distilbert-f16.gguf distilbert-greedy.gguf distilbert-sa.gguf distilbert-hybrid.gguf
python src/evaluation/build_experiment_registry.py
```

Current tracked GGUF benchmark status:

- canonical CPU/CUDA batch: `src/llama.cpp/benchmarks/20260404_134936_*`
- archived non-canonical batch: `src/llama.cpp/benchmarks/20260404_123103_*`
- the archived batch is excluded from CUDA comparisons because `llama.cpp` reported no usable GPU support and ignored GPU offload there
- `src/results/gguf_runtime_results.json` now records benchmark summaries for `Primary Baseline (FP16 GGUF)`, `Greedy Mixed`, `SA (v1)`, and `Hybrid SA (ours)`

Use these results only for supplementary deployment/runtime discussion, not as replacements for:

- SST-2 accuracy
- F1
- ROC-AUC
- ONNX-based classifier evaluation

## 8. Recommended Command Order

If you want the complete current experiment package:

```powershell
python src/training/realistic/train_baseline_model.py
python src/training/realistic/train_frozen_model.py
python src/training/controlled/train_baseline_model.py
python src/training/controlled/train_frozen_model.py
python src/evaluation/evaluate_reference_models.py
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
python src/analysis/plot_training_convergence.py
python src/evaluation/build_experiment_registry.py
```

If the models already exist and you only want to refresh results:

```powershell
python src/evaluation/evaluate_reference_models.py
python src/evaluation/evaluate_all_onnx_models.py
python src/analysis/compare_model_sizes.py
python src/evaluation/build_experiment_registry.py
python src/analysis/generate_paper_figures.py
```

## 9. What Must Exist At The End

### Models

- checkpoints in `src/models/realistic/` and `src/models/controlled/`
- ONNX exports in `src/models/`

### Results

- reference checkpoint result file
- ONNX result file
- size comparison result file
- experiment registry
- supplementary GGUF runtime placeholder/result file
- training-history JSON files for realistic and controlled runs

### Figures

- SA convergence and Pareto frontier figures
- final comparison plots
- training convergence plot against cumulative training steps

## 10. Practical Interpretation For The Paper

The most defensible framing is:

- `ONNX` provides the main scientific evidence for mixed-precision sensitivity-aware optimization.
- `llama.cpp` / `GGUF` provides supplementary runtime evidence only.
- `FP16` is the deployment-oriented baseline.
- one `FP32` baseline is retained as a research reference.
- `FP8` is an optional extension track only for the three mixed-precision optimized models.

That gives you:

- a clear primary evaluation story
- a realistic deployment baseline
- a defensible reference point
- room for future FP8 experiments without destabilizing the current pipeline

