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
- Optional future FP8 track: `Greedy Mixed`, `SA Mixed (v1)`, `Hybrid INT8+SA (Ours)`

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
