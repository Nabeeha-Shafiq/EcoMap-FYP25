# EcoMap MLP — Teacher Training Branch

> **Branch:** `teacher` | **Purpose:** Fuse multi-modal embeddings (UNI, scVI, RCTD) and train an MLP classifier with spatial validation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Quick Start](#2-quick-start)
3. [Dataset Requirements](#3-dataset-requirements)
4. [Configuration](#4-configuration)
5. [Key Features](#5-key-features)
6. [Ablation Studies](#6-ablation-studies)
7. [Output Structure](#7-output-structure)
8. [Reproducing on a New Dataset](#8-reproducing-on-a-new-dataset)

---

## 1. Pipeline Overview

| Stage | Script | Output |
|-------|--------|--------|
| **1** | `load_input_embeddings.py` | Validated embeddings in NumPy format |
| **2** | `validate_initial_embeddings.py` | QC reports, correlation matrices |
| **3** | `preprocess_embeddings.py` | PCA-reduced fused embeddings |
| **3.5** | `validate_and_visualize_preprocessing.py` | Spatial heatmaps, preprocessing validation |
| **4** | `train_mlp.py` | Trained model, 5-fold CV metrics, predictions |
| **6** | `create_spatial_visualizations.py` | Patient-specific spatial maps, 3D landscapes |

---

## 2. Quick Start

### Step 1 — Set up the environment

```bash
source ../.venv/bin/activate
mkdir -p data/
```

### Step 2 — Download the datasets

Three reference datasets are available here:

> [Google Drive — Dataset Folder](https://drive.google.com/drive/folders/1h2mY0to3B52E_IKbG4DPKqhckzK-08o-?usp=sharing)

Unzip the folders into your **project root**, then point the config file at the relevant data directories.

### Step 3 — Place your data files

Put CSV files in `data/input_dataset/` — see [Dataset Requirements](#3-dataset-requirements) for the full list.

### Step 4 — Edit the config

Open `config/modular_flexible.yaml` and set:
- Input/output directory paths
- PCA variance threshold (default: `0.95`)
- Training parameters (learning rate, epochs, batch size)

### Step 5 — Run

```bash
bash run_pipeline.sh config/modular_flexible.yaml
```

---

## 3. Dataset Requirements

Place these files in `data/input_dataset/`:

| File | Description |
|------|-------------|
| `image_encoder_embeddings.csv` | 1024D UNI image embeddings |
| `gene_embeddings.csv` | 128D scVI gene expression embeddings |
| `cell_encoder_embeddings.csv` | 25D RCTD cell type composition embeddings |
| `barcode_labels.csv` | Labels (ecotype, tumor_subtype, etc.) |
| `barcode_metadata.csv` | Spatial metadata — must include `barcode`, `patient_id`, `x_coord`, `y_coord`, `ecotype` |

> `x_coord` and `y_coord` in `barcode_metadata.csv` are required for spatial visualizations. Missing columns will cause Stage 3.5 and Stage 6 to fail.

---

## 4. Configuration

All parameters are controlled through `config/modular_flexible.yaml`, no changes to script code are needed.

Key parameters to review before running:

```yaml
pca:
  variance_ratio: 0.95   # Proportion of variance retained after PCA reduction

training:
  learning_rate: 0.001
  epochs: 200
  batch_size: 32

paths:
  input_dir: data/input_dataset/
  output_dir: results/
```

---

## 5. Key Features

- **Fully modular** — each stage is an independent Python script; individual stages can be re-run without restarting the full pipeline
- **Config-driven** — all parameters live in YAML; no hardcoded values in scripts
- **Reproducible** — deterministic seeds, all metrics saved to disk
- **Spatial-aware** — predictions validated against tissue coordinates throughout
- **Organized outputs** — structured results folder with separate preprocessing, training, and post-training directories

### Project File Reference

| File / Folder | Purpose |
|---------------|---------|
| `run_pipeline.sh` | Pipeline orchestrator |
| `pipeline/` | 6 modular stage scripts |
| `config/modular_flexible.yaml` | All configuration |
| `metrics_tracker.py` | Training utilities |
| `post_training_visualizations.py` | Post-training analysis tools |

---

## 6. Ablation Studies

To test different configurations without overwriting previous results, change `output_dir` in the config and adjust the parameters you want to vary:

```bash
# In config/modular_flexible.yaml, adjust:
#   pca.variance_ratio: 0.90 / 0.95 / 0.99
#   training.learning_rate, batch_size, epochs
#   paths.output_dir: "results/ablation_pca90" (or similar)

bash run_pipeline.sh config/modular_flexible.yaml
```

Each run writes to its own output folder, keeping results from different configurations separate and comparable.

---

## 7. Output Structure

```
results/
├── preprocessing/
│   ├── metrics/            # QC reports, validation CSVs
│   └── visualizations/     # Correlation plots, spatial heatmaps
│
├── training/
│   ├── metrics/            # training_results.json, predictions, model weights
│   └── visualizations/     # Confusion matrix, training curves, accuracy plots
│
└── post-training/
    ├── metrics/            # Post-training evaluation
    └── visualizations/     # Ecotype maps, confidence heatmaps, spatial localisation plots

```

---

## 8. Reproducing on a New Dataset

The pipeline is designed to run on any compatible dataset without touching script code:

1. **Replace CSVs** in `data/input_dataset/` with your files (matching the column structure in [Dataset Requirements](#3-dataset-requirements))
2. **Edit config** — update `input_dir`, `output_dir`, and any training parameters
3. **Run the pipeline**, all stages execute automatically
4. **Check results** in your specified output folder
