# Modular Unified Teacher-Student Pipeline

**Branch**: `student`  
**Purpose**: End-to-end knowledge distillation pipeline for training compact student models using teacher knowledge  

---

## Quick Start

### Train Unified Teacher-Student Model (One Command)

**For GEO Dataset:**
```bash
./run_unified_pipeline.sh config/modular_unified_teacher_student_GEO.yaml
```

**For ZENODO Dataset:**
```bash
./run_unified_pipeline.sh config/modular_unified_teacher_student_Zenodo.yaml
```

That's it! One command handles everything:
1. **STAGE 0**: Load input embeddings (UNI, scVI, cell composition)
2. **STAGE 0.5**: Apply PCA preprocessing and feature reduction
3. **PHASE 1**: Train teacher model (multimodal, 5-fold cross-validation)
4. **PHASE 2**: Build ensemble teacher from all folds
5. **PHASE 3**: Train student model with knowledge distillation
6. **PHASE 4**: Generate comprehensive visualizations (spatial, confidence, neighborhood analysis)

---

## Configuration

Three config files are provided:

- **`config/modular_unified_teacher_student_flexible.yaml`** - Template for custom datasets
- **`config/modular_unified_teacher_student_GEO.yaml`** - Pre-configured for GEO dataset
- **`config/modular_unified_teacher_student_Zenodo.yaml`** - Pre-configured for ZENODO dataset

To use a custom dataset, modify `modular_unified_teacher_student_flexible.yaml` with your dataset paths and run:
```bash
./run_unified_pipeline.sh config/modular_unified_teacher_student_flexible.yaml
```

---

## Model Architecture

### Teacher Model (Multimodal)
- **Input**: 171D (UNI:150D + scVI:128D + Cell Composition:25D for GEO, 15D for ZENODO)
- **Hidden layers**: [256, 128, 64]
- **Output**: 5-class ecotype predictions
- **Accuracy**: ~91%

### Student Model (Morphology-Only)
- **Input**: 18D (UNI embeddings, PCA reduced from 1024D)
- **Hidden layers**: [128, 64, 32]
- **Output**: 5-class ecotype predictions
- **Accuracy**: ~56%
- **Advantage**: 95% parameter reduction, human-interpretable morphology features

---

## Output Structure

Results are organized by dataset:

```
[GEO/ZENODO] Ablation Study/
├── TEACHER_UNIFIED_[DATASET]_0.6_PCA_ver5/
│   ├── training/metrics/       # Cross-validation metrics, fold results
│   ├── training/models/        # Fold-specific teacher models
│   └── .working/               # Preprocessed embeddings, PCA models
│
└── STUDENT_UNIFIED_[DATASET]_0.6_PCA_ver5/
    ├── training/metrics/       # Student accuracy, F1, predictions CSV
    ├── training/models/        # Final student model weights
    ├── post-training/visualizations/
    │   ├── spatial/           # Ecotype maps with tissue coordinates
    │   ├── spatial_confidence/ # Prediction confidence heatmaps
    │   ├── spatial_comparison/ # Teacher vs student predictions
    │   ├── neighborhood/      # Neighbor composition analysis
    │   └── preprocessing/     # Embedding quality metrics
    └── pipeline_execution.log  # Complete execution log
```

---

## Dataset Requirements

Each dataset needs:
- `barcode_labels.csv` - Sample labels (one per row)
- `barcode_metadata.csv` - Metadata including x_coord, y_coord, patient_id for spatial visualization
- `label_mapping.json` - Class name to index mapping
- `UNI_EMBEDDINGS_COMBINED.CSV` - 1024D morphology embeddings (student input)
- `SCVI_EMBEDDINGS_COMBINED.CSV` - 128D gene expression embeddings (teacher only)
- `CELL_COMPOSITION_EMBEDDINGS*.CSV` - Cell type composition (teacher only)

---

## Key Features

✅ **One-command execution** - Full pipeline from raw embeddings to visualizations  
✅ **Automatic PCA sharing** - Teacher learns PCA models, student reuses them  
✅ **Cross-validation** - 5-fold stratified CV for robust evaluation  
✅ **Knowledge distillation** - Soft label + feature matching loss (α=0.7, β=0.2, γ=0.1)  
✅ **Spatial visualizations** - Map predictions onto tissue coordinates  
✅ **Publication-ready figures** - Per-patient metrics, confidence maps, neighborhood analysis  
✅ **Comprehensive logging** - Track model performance across all training phases  

---

## Troubleshooting

**Q: Pipeline fails during teacher training?**  
A: Check that all input embedding CSV files exist and have the correct column names.

**Q: Student model accuracy is low?**  
A: This is expected (~56% vs teacher ~91%). Student uses only morphology while teacher uses multimodal data. The 35% gap reflects information lost.

**Q: Visualizations not generated?**  
A: Ensure `barcode_metadata.csv` contains `x_coord` and `y_coord` columns. Other visualizations generate regardless.

---

## Key Hyperparameters

Adjust in YAML configs:

```yaml
teacher:
  n_epochs: 200
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 20

student:
  n_epochs: 150
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 15
  
distillation:
  alpha: 0.7        # Hard target weight
  beta: 0.2         # Soft target weight
  gamma: 0.1        # Feature matching weight
  temperature: 4.0  # Teacher logit softness

embeddings:
  image_encoder:
    pca_variance: 0.6  # Reduce 1024D to ~150D
```

---

**Status**: March 2026 | Production Ready | STUDENT BRANCH
