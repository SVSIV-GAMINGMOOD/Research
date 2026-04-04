# Source Layout

- `artifacts/`: generated text and CSV outputs that are useful to keep with the project
- `analysis/`: size studies, sensitivity analysis, and figure-generation scripts
- `docs/`: project notes and workflow documentation
- `evaluation/`: baseline, frozen, INT8, and comparison evaluation entry points
- `export/`: ONNX export, ONNX quantization, and Hugging Face export scripts
- `figures/`: generated plots and thesis/paper figures
- `models/`: trained checkpoints and exported model files
- `optimization/`: greedy and simulated-annealing search logic
- `results/`: JSON and text evaluation summaries
- `shared/`: reusable helpers shared by training, evaluation, and export scripts
- `training/`: model training entry points, including separate `realistic/` and `controlled/` regimens

Current experiment policy:

- Primary metrics backend: `ONNX`
- Supplementary runtime backend: `llama.cpp` / `GGUF`
- Primary deployment baseline: `FP16`
- Research reference: `FP32`
- Optional future FP8 track: `Greedy Mixed`, `SA (v1)`, `Hybrid SA (ours)`

Training checkpoint selection:

- Downstream checkpoint consumers now read from `config/experiment_settings.json`
- `training.active_regimen` selects which regimen-backed checkpoints are used for exports and evaluations
- Current default: `realistic`
- Regimen checkpoints live under `models/realistic/` and `models/controlled/`
- Exported ONNX and quantized artifacts still live at `models/` root unless a script explicitly writes elsewhere

Canonical experiment outputs live in `results/`, especially:

- `reference_model_results.json`
- `all_model_results.json`
- `size_comparison.json`
- `experiment_registry.json`
- `gguf_runtime_results.json`
- `training_histories/realistic/*.json`
- `training_histories/controlled/*.json`

Supplementary GGUF workflow:

- export the cleaned DistilBERT backbone:
  - `python src/export/export_baseline_to_huggingface.py`
- generate tensor maps for mixed-precision presets:
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 greedy`
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 sa`
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/generate_tensor_map.ps1 hybrid`
- convert to GGUF with the preset wrapper:
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 greedy`
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 sa`
  - `powershell -ExecutionPolicy Bypass -File src/llama.cpp/convert_distilbert_gguf.ps1 hybrid`
- run CPU and CUDA runtime benchmarks:
  - `bash src/llama.cpp/benchmark_distilbert_embeddings.sh distilbert-f16.gguf distilbert-greedy.gguf distilbert-sa.gguf distilbert-hybrid.gguf`
- consolidate benchmark outputs into the tracked registry:
  - `python src/evaluation/build_experiment_registry.py`
- write supplementary runtime measurements to:
  - `src/results/gguf_runtime_results.json`

Tracked GGUF status:

- `src/results/gguf_runtime_results.json` now records CPU and CUDA benchmark summaries for `FP16`, `Greedy Mixed`, `SA (v1)`, and `Hybrid SA (ours)`
- canonical CPU/CUDA run batch: `src/llama.cpp/benchmarks/20260404_134936_*`
- archived non-canonical batch: `src/llama.cpp/benchmarks/20260404_123103_*` because `llama.cpp` reported no usable GPU support and ignored GPU offload there

Keep GGUF results as supplementary runtime evidence only; use ONNX as the main source for classifier-faithful metrics.

Training policy:

- `training/realistic/`
  - deployment-oriented setting
  - `max_epochs = 20`
  - early stopping with `patience = 2`
  - reports the best validation checkpoint
- `training/controlled/`
  - fixed-length comparison setting
  - `10` epochs
  - no early stopping
- `analysis/plot_training_convergence.py`
  - plots validation accuracy against cumulative training steps from the saved history JSON files

Protected folders left unchanged:
- `llama.cpp/`
- `models/`

