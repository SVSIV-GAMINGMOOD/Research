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
- `shared/`: reusable helpers shared by evaluation and export scripts
- `training/`: model training entry points

Protected folders left unchanged:
- `llama.cpp/`
- `models/`
