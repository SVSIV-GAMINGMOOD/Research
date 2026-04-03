# Documentation Guide

This folder contains the current experiment documentation for the project.

## Primary documents

- [roadmap.md](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\docs\roadmap.md)
  - the main execution and experiment policy document
  - explains the full run order
  - describes result files and experiment tracking

## Current experiment policy

- Primary research metrics backend:
  - `ONNX`
- Supplementary runtime backend:
  - `llama.cpp` / `GGUF`
- Primary deployment baseline:
  - `FP16`
- Research reference:
  - `FP32`
- Optional future precision branch only:
  - `Greedy Mixed`
  - `SA Mixed (v1)`
  - `Hybrid INT8+SA (Ours)`

## Result tracking

The canonical experiment outputs live in:

- [results/README.md](c:\Users\skvar\Desktop\workspace\Research\edge_bert\src\results\README.md)

The main consolidated result files are:

- `src/results/reference_model_results.json`
- `src/results/all_model_results.json`
- `src/results/size_comparison.json`
- `src/results/experiment_registry.json`
- `src/results/gguf_runtime_results.json`

Training-history outputs for the two training regimens live in:

- `src/results/training_histories/realistic/`
- `src/results/training_histories/controlled/`

New training workflow summary:

- `src/training/realistic/`
  - `max_epochs = 20`
  - early stopping with `patience = 2`
  - best-checkpoint reporting
- `src/training/controlled/`
  - fixed `10` epochs
  - no early stopping
- `src/analysis/plot_training_convergence.py`
  - plots validation accuracy versus cumulative training steps across saved training histories
