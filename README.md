# Brain-Robustness

Benchmarking out-of-distribution robustness across deep vision architectures and exploring alignment with neural representations (Brain-Score & THINGS-EEG2).

---

## Overview

This project analyzes the robustness of vision models across clean ImageNet and multiple distribution-shift benchmarks:
- **Clean ImageNet**: Standard validation accuracy.
- **ImageNet-Sketch**: Robustness to abstract sketches and line drawings.
- **ImageNet-A**: Natural adversarial examples designed to fool standard classifiers.
- **ImageNet-R**: Renditions (art, cartoons, deviant textures, sculptures).

Furthermore, the project investigates how these robustness behaviors relate to biological neural representations (IT cortex alignments via **Brain-Score** and event-related potentials in **THINGS-EEG2**).

---

## Directory Structure

```text
brain-robustness/
├── data/
│   ├── robustness/          # Evaluation CSVs (ImageNet, Sketch, ImageNet-A, ImageNet-R)
│   └── things_eeg2/         # THINGS-EEG2 dataset instructions & download pointers
├── notebooks/               # Interactive exploration & analysis notebooks
├── results/                 # Merged regression tables & scatter plots
├── src/
│   ├── smoke_test.py        # Validates model loading, forward pass & data files
│   ├── build_robustness_table.py # Merges benchmarks, runs logit OLS regressions, produces plots
│   └── brainscore_test.py   # Brain-Score benchmark evaluation pipeline
├── .gitignore               # Strict ignore rules for large data, weights & venv
├── requirements.txt         # Pinned python environment dependencies
└── README.md
```

---

## Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or using standard python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Smoke Test

Verify your environment, PyTorch, timm model weights, and benchmark data:

```bash
python src/smoke_test.py
```

### 3. Build Robustness Table & Regressions

Fit logit OLS regressions, compute robustness residuals, and generate summary scatter plots:

```bash
python src/build_robustness_table.py
```

Outputs will be saved in `results/`:
- `results/robustness_table.csv`
- `results/scatter_imagenet_sketch.png`
- `results/scatter_imagenet_a.png`
- `results/scatter_imagenet_r.png`

---

## Robustness Regression Summary

| Benchmark | Slope | Intercept | R² |
|---|---|---|---|
| **ImageNet-Sketch** | 1.3403 | −2.6918 | **0.827** |
| **ImageNet-A** | 4.1131 | −7.4748 | **0.871** |
| **ImageNet-R** | 1.3109 | −2.1147 | **0.741** |

> All three benchmarks are strongly predicted by clean accuracy ($R^2 > 0.74$), with residual analysis highlighting architectures exhibiting outsized robustness.
