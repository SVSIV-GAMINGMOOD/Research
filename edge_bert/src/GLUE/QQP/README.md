# GLUE QQP Pipeline

This folder contains a fully isolated QQP pipeline. It trains fresh QQP checkpoints from `distilbert-base-uncased` and keeps all models, figures, configs, and results inside `src/GLUE/QQP/`.

Task-local settings live in `src/GLUE/QQP/config/experiment_settings.json`. Update that file when you want to change QQP-specific training, evaluation, sensitivity, search, runtime, or output settings without touching SST-2 or other GLUE tasks.

Primary QQP model lineup:

- `FP32 Research Reference`
- `Primary Baseline (FP16)`
- `INT8 Uniform`
- `Greedy Mixed`
- `SA (v1)`
- `Hybrid SA (ours)`

Recommended command order from the repo root:

```powershell
python src/GLUE/QQP/training/realistic/train_baseline_model.py
python src/GLUE/QQP/training/controlled/train_baseline_model.py
python src/GLUE/QQP/evaluation/evaluate_reference_models.py
python src/GLUE/QQP/export/export_baseline_checkpoint_to_onnx.py
python src/GLUE/QQP/export/export_fp16_onnx.py
python src/GLUE/QQP/export/quantize_baseline_onnx_to_int8.py
python src/GLUE/QQP/analysis/analyze_layer_sensitivity.py
python src/GLUE/QQP/optimization/generate_greedy_bit_config.py
python src/GLUE/QQP/optimization/run_simulated_annealing_search.py
python src/GLUE/QQP/export/export_greedy_quantized_checkpoint.py
python src/GLUE/QQP/export/export_greedy_checkpoint_to_onnx.py
python src/GLUE/QQP/export/export_sa_quantized_onnx.py
python src/GLUE/QQP/export/export_hybrid_quantized_onnx.py
python src/GLUE/QQP/evaluation/evaluate_all_onnx_models.py
python src/GLUE/QQP/analysis/compare_model_sizes.py
python src/GLUE/QQP/analysis/plot_accuracy_size_tradeoff.py
python src/GLUE/QQP/analysis/generate_paper_figures.py
```

Notes:

- Sensitivity analysis uses a deterministic random subset of 2048 validation examples.
- The `realistic` regimen is the default source for sensitivity, export, and mixed-precision search.
- Simulated annealing follows the lightweight original search pattern: baseline accuracy comes from the QQP PyTorch checkpoint, while size and latency normalization come from cached reference metrics and configured fallbacks.
- FP16 ONNX remains part of the evaluation and reporting flow, not the SA search setup path.
- The latency comparison figure excludes `Primary Baseline (FP16)` because its runtime path is not directly comparable to the CPU ONNX latency bars used for the other deployment models.
