# Modular Embedding Fusion & MLP Classification Pipeline

A reproducible pipeline for fusing multi-modal embeddings (UNI, scVI, RCTD) and training an MLP classifier with spatial validation.

## Quick Start

### 1. Setup
```bash
# Activate environment
source ../.venv/bin/activate

# Create data directory
mkdir -p data/input_dataset data/arrays
```

### 2. Prepare Data
Place CSV files in `data/input_dataset/`:
- `uni_embeddings.csv` - UNI image embeddings (1024 dimensions)
- `scvi_embeddings.csv` - scVI gene embeddings (128 dimensions)  
- `rctd_embeddings.csv` - RCTD cell type embeddings (25 dimensions)
- `barcode_labels.csv` - Labels (ecotype, tumor_subtype, etc.)
- `barcode_metadata.csv` - Spatial coordinates (barcode, patient_id, x_coord, y_coord, ecotype)

### 3. Configure
Edit `config/modular_flexible.yaml`:
- Set input/output directory paths
- Adjust PCA variance threshold (default: 0.95)
- Modify training parameters (learning rate, epochs, batch size)

### 4. Run
```bash
bash run_pipeline.sh config/modular_flexible.yaml
```

## Pipeline Stages

| Stage | Script | Output |
|-------|--------|--------|
| 1 | `load_input_embeddings.py` | Validated embeddings in numpy format |
| 2 | `validate_initial_embeddings.py` | QC reports, correlation matrices |
| 3 | `preprocess_embeddings.py` | PCA-reduced fused embeddings |
| 3.5 | `validate_and_visualize_preprocessing.py` | Spatial heatmaps, preprocessing validation |
| 4 | `train_mlp.py` | Model, 5-fold CV metrics, predictions |
| 6 | `create_spatial_visualizations.py` | Patient-specific spatial maps, 3D landscapes |

## Output Structure

```
results/
├── preprocessing/
│   ├── metrics/           → QC reports, validation CSVs
│   └── visualizations/    → Correlation plots, spatial heatmaps
├── training/
│   ├── metrics/           → training_results.json, predictions, model weights
│   └── visualizations/    → Confusion matrix, training curves, accuracy plots
└── post-training/
    ├── metrics/           → Post-training evaluation
    └── visualizations/    → Ecotype maps, confidence heatmaps, 3D plots
```

## Key Features

- **Fully modular**: Each stage is independent Python script
- **Configurable**: All parameters in YAML (no code changes needed)
- **Reproducible**: Deterministic seeds, complete metrics saved
- **Spatial-aware**: Validates predictions against tissue coordinates
- **Production-ready**: Error handling, logging, organized outputs

## Files

- `run_pipeline.sh` - Orchestrator
- `pipeline/` - 6 modular stage scripts
- `config/modular_flexible.yaml` - Configuration
- `metrics_tracker.py` - Training utilities
- `post_training_visualizations.py` - Analysis tools

## Ablation Studies

To run different configurations:
```bash
# Edit config/modular_flexible.yaml:
# - Change pca.variance_ratio (0.90, 0.95, 0.99, etc.)
# - Try different learning rates, batch sizes, epochs
# - Point output_dir to different folder

export OUTPUT_DIR="ablation_test_1"
bash run_pipeline.sh config/modular_flexible.yaml
```

## Reproducibility

This pipeline is designed to be shared and reproduced on new datasets. To use on your data:

1. **Replace CSVs** in `data/input_dataset/`
2. **Edit config** with your paths and parameters  
3. **Run pipeline** - Everything else is automatic
4. **Check results** in output folder

No modifications to script code needed.

