# Results Folder

This folder is the canonical store for experiment outputs and result registries.

## Primary result files

- `reference_model_results.json`: checkpoint-level PyTorch reference measurements
- `all_model_results.json`: ONNX evaluation metrics for the main model comparison table
- `size_comparison.json`: theoretical and exported size comparison across model variants
- `experiment_registry.json`: consolidated registry of created models, result sources, and experiment tracks
- `gguf_runtime_results.json`: supplementary `llama.cpp` / GGUF CPU and CUDA runtime summaries, benchmark provenance, and archived non-canonical runs

## Supporting result files

- `sensitivity_results.json`: layer sensitivity analysis output when generated
- `sa_search_results.json`: constrained Pareto simulated annealing search output when generated
- `sa_best_config.json`: default representative SA configuration when generated
- `training_histories/realistic/*.json`: epoch-by-epoch histories for realistic training runs
- `training_histories/controlled/*.json`: epoch-by-epoch histories for controlled training runs

## Two-track evaluation policy

- Primary research metrics:
  - ONNX-based evaluation for classification quality, latency, and model-size comparisons
- Supplementary runtime benchmarking:
  - `llama.cpp` / GGUF benchmarking for runtime-only comparisons after conversion fidelity is validated
  - canonical tracked CPU/CUDA batch currently comes from `src/llama.cpp/benchmarks/20260404_134936_*`
  - archived `20260404_123103_*` runs are retained for provenance only because GPU offload was not actually available in that batch

## Precision policy

- Primary deployment baseline: `FP16`
- Research reference: `FP32`
- Optional experimental track only:
  - `Greedy Mixed`
  - `SA (v1)`
  - `Hybrid SA (ours)`

The FP8 track is intentionally optional and is not required for the current mandatory result set.

## Training-history tracking

The training-history JSON files now store:

- cumulative training steps per epoch
- validation accuracy and F1 per epoch
- best epoch and best global step
- total validation passes
- total validation examples processed

This makes it easier to document the validation budget for:

- realistic training with early stopping
- controlled fixed-epoch comparisons
- convergence plots based on accuracy versus training steps

