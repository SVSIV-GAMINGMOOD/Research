# GLUE MRPC Pipeline

This folder contains a fully isolated MRPC pipeline. It trains fresh MRPC checkpoints from `distilbert-base-uncased` and keeps all models, figures, configs, and results inside `src/GLUE/MRPC/`.

Task-local settings live in `src/GLUE/MRPC/config/experiment_settings.json`. Update that file when you want to change MRPC-specific training, evaluation, sensitivity, search, runtime, or output settings without touching SST-2 or other GLUE tasks.

Primary MRPC model lineup:

- `FP32 Research Reference`
- `Primary Baseline (FP16)`
- `INT8 Uniform`
- `Greedy Mixed`
- `SA (v1)`
- `Hybrid SA (ours)`

Recommended command order from the repo root:

```powershell
python src/GLUE/MRPC/training/realistic/train_baseline_model.py
python src/GLUE/MRPC/training/controlled/train_baseline_model.py
python src/GLUE/MRPC/evaluation/evaluate_reference_models.py
python src/GLUE/MRPC/export/export_baseline_checkpoint_to_onnx.py
python src/GLUE/MRPC/export/export_fp16_onnx.py
python src/GLUE/MRPC/export/quantize_baseline_onnx_to_int8.py
python src/GLUE/MRPC/analysis/analyze_layer_sensitivity.py
python src/GLUE/MRPC/optimization/generate_greedy_bit_config.py
python src/GLUE/MRPC/optimization/run_simulated_annealing_search.py
python src/GLUE/MRPC/export/export_greedy_quantized_checkpoint.py
python src/GLUE/MRPC/export/export_greedy_checkpoint_to_onnx.py
python src/GLUE/MRPC/export/export_sa_quantized_onnx.py
python src/GLUE/MRPC/export/export_hybrid_quantized_onnx.py
python src/GLUE/MRPC/evaluation/evaluate_all_onnx_models.py
python src/GLUE/MRPC/analysis/compare_model_sizes.py
python src/GLUE/MRPC/analysis/plot_accuracy_size_tradeoff.py
python src/GLUE/MRPC/analysis/generate_paper_figures.py
```

Notes:

- Sensitivity analysis uses a deterministic random subset of 2048 validation examples.
- The `realistic` regimen is the default source for sensitivity, export, and mixed-precision search.
- Simulated annealing follows the lightweight original search pattern: baseline accuracy comes from the MRPC PyTorch checkpoint, while size and latency normalization come from cached reference metrics and configured fallbacks.
- FP16 ONNX remains part of the evaluation and reporting flow, not the SA search setup path.
- The latency comparison figure excludes `Primary Baseline (FP16)` because its runtime path is not directly comparable to the CPU ONNX latency bars used for the other deployment models.
