
---

# 🚧 2. ROADMAP.md (What You’ve Done vs What’s Missing)

This is **EXTREMELY useful for research + planning + paper writing**.

```markdown
# 🚧 Research Roadmap: Mixed Precision Quantization

This document summarizes the current state of the project and outlines remaining work required for a **publication-ready system**.

---

## ✅ Completed Work

### 🧠 Model & Training
- Baseline DistilBERT training (SST-2)
- Frozen model (FAR: partial layer freezing)
- Evaluation pipeline (accuracy, F1, ROC-AUC)

---

### ⚙️ Quantization
- INT8 ONNX quantization (baseline + frozen)
- Mixed precision quantization:
  - FP8 / FP6 / FP4
- Custom quantized PyTorch layers
- ONNX export for all models

---

### 🔍 Sensitivity Analysis
- Layer-wise sensitivity using 4-bit simulation
- Accuracy-based robustness metric
- Used to guide bitwidth allocation

---

### 🧩 Optimization Framework
- Multi-objective energy function:
  - Accuracy
  - Latency
  - Model size

- Greedy bitwidth allocation (baseline)

---

### 🔥 Simulated Annealing (Core Contribution)
- Global optimization over bitwidth assignments
- Supports heterogeneous precision
- Produces improved trade-offs vs greedy

---

### ⚡ Hybrid Quantization
- INT8 embeddings + SA-based mixed precision
- Improved balance of:
  - accuracy
  - latency
  - size

---

### 📊 Evaluation & Visualization
- Unified evaluation across 7 model variants
- Tradeoff curves (accuracy vs size)
- Bit allocation heatmaps
- Latency comparison plots

---

## ⚠️ Limitations (Current Gaps)

### 🔴 Quantization Limitations
- Only **weight quantization**
- No **activation quantization**
- FP computation still used (no true low-bit inference)

---

### 🔴 Sensitivity Analysis
- Only evaluated at **4-bit**
- Does not cover:
  - FP6
  - FP8
- No activation sensitivity

---

### 🔴 Optimization
- Only compared against **Greedy**
- No comparison with:
  - Random search
  - Hill climbing
  - ILP / RL methods

---

### 🔴 Efficiency
- Full model evaluation per iteration (slow)
- Deepcopy in energy function
- No caching / reuse

---

### 🔴 Latency Modeling
- Simplified estimation
- Not hardware-calibrated
- Assumes linear scaling with bitwidth

---

## 🚀 Planned Improvements

### 🧠 Sensitivity Analysis
- Multi-bit sensitivity (4, 6, 8)
- Activation sensitivity
- Normalized sensitivity metric

---

### ⚙️ Optimization Enhancements
- Add:
  - Random search baseline
  - Hill climbing
- Convergence plots (SA vs Greedy)
- Multi-run statistics (mean/std)

---

### 🔥 Algorithmic Improvements
- Constraint-based optimization:
  - Accuracy ≥ threshold
- Hybrid SQ + SA approach
- Improved neighbor generation in SA

---

### ⚡ Quantization Improvements
- Activation quantization
- Per-channel quantization
- Better clipping (percentile-based)

---

### 📊 Evaluation Improvements
- Pareto frontier analysis
- Ablation studies:
  - No sensitivity
  - No SA
  - Full pipeline
- Cross-task generalization (GLUE tasks)

---

### 🖥️ System Improvements
- Faster evaluation (subset / cached logits)
- Reduced deepcopy overhead
- Modular quantization pipeline

---

### 🔗 Deployment
- Better ONNX optimization
- Optional hardware benchmarking

---

## 🎯 Target Outcome

A **publication-ready system** demonstrating:

- Sensitivity-aware mixed precision quantization
- Global optimization via simulated annealing
- Superior trade-offs vs greedy baseline
- Applicability to transformer architectures

---

## 🧠 Final Goal

> Develop a general framework for **efficient transformer compression** using **stochastic optimization and mixed precision quantization**.