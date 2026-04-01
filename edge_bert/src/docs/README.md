# DistilBERT Quantization and Evaluation Framework

This repository contains a comprehensive framework for quantizing and evaluating DistilBERT models on the SST-2 dataset (GLUE benchmark). It includes various quantization techniques such as:
- FP32 baseline
- INT8 uniform quantization
- FAR (frozen layer) quantization
- Greedy mixed precision
- Simulated Annealing (SA) mixed precision
- Hybrid INT8+SA (our proposed method)

The framework is designed to compare accuracy, model size, and latency across different quantization strategies.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Results and Outputs](#results-and-outputs)
- [Notes](#notes)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install torch transformers onnxruntime datasets scikit-learn matplotlib codecarbon psutil
   ```

## Project Structure

