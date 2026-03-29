# Student Model - Knowledge Distillation Pipeline

**Branch**: `student`  
**Purpose**: Complete modular knowledge distillation pipeline for training student models (morphology-only) using teacher knowledge (multimodal)  
**Status**: ✅ Production Ready

---

## Quick Start (One Command)

### Train Complete Student Pipeline (GEO Dataset)
```bash
./run_unified_pipeline.sh config/modular_GEO.yaml
```

### Train Complete Student Pipeline (ZENODO Dataset)
```bash
./run_unified_pipeline.sh config/modular_Zenodo.yaml
```

That's it! One command runs: **Teacher → Ensemble → Student** automatically.

---

## What Gets Trained

| Phase | Input | Output | Time |
|-------|-------|--------|------|
| **1. Teacher** | Multimodal embeddings (UNI+scVI+RCTD) | 5-fold teacher models | ~16 min |
| **2. Ensemble** | All 5 teacher fold models | Averaged ensemble teacher | ~1 min |
| **3. Student** | Morphology-only (UNI), teacher labels | 5-fold student models | ~25 min |

**Total**: ~42 minutes

---

## Configuration System

### Modular YAML Approach
All configuration is **YAML-driven**. No code changes needed for different datasets or ablations.

#### Files Structure
```
config/
├── modular_GEO.yaml              ← Teacher config (multimodal)
├── modular_Student_GEO.yaml      ← Student config (morphology-only)
├── modular_Zenodo.yaml           ← Teacher config for ZENODO
├── modular_Student_Zenodo.yaml   ← Student config for ZENODO
└── modular_flexible.yaml         ← Template for custom ablations
```

---

## How to Customize: Step-by-Step Guide

### Step 1: Edit Student Config YAML

Open `config/modular_Student_GEO.yaml`:

```yaml
# Section 1: Dataset & Input Paths
input_dataset:
  labels_file: "./GEO data/barcode_labels.csv"          # ← Change dataset path
  image_encoder_embeddings: "./GEO data/uni_embeddings_1024d_combined.csv"

# Section 2: Feature Preprocessing
embeddings:
  image_encoder:
    pca_variance: 0.6                                    # ← Change PCA variance
    normalize: true

# Section 3: Student Architecture
training:
  hidden_dims: [128, 64, 32]                             # ← Change layer sizes
  learning_rate: 0.001                                   # ← Change LR
  batch_size: 32                                         # ← Change batch size
  n_epochs: 150                                          # ← Change max epochs
  early_stopping_patience: 15                            # ← Change patience

# Section 4: Knowledge Distillation Parameters
distillation:
  alpha: 0.7                                             # ← Hard loss weight
  beta: 0.2                                              # ← Soft loss weight
  gamma: 0.1                                             # ← Feature loss weight
  temperature: 4.0                                       # ← Softmax temperature

# Section 5: Output Directory
output:
  output_dir: "./GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA"  # ← Output path
```

### Step 2: Update Teacher Path (if different)

```yaml
distillation:
  teacher_model_path: "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA/teacher_model_FROZEN.pt"
  teacher_preprocessed_dir: "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA/.working/preprocessed_arrays"
```

### Step 3: Run Pipeline

```bash
./run_unified_pipeline.sh config/modular_Student_GEO.yaml
```

---

## Ablation Study Examples

### Example 1: Ablate PCA Variance

```yaml
# config/modular_Student_GEO_PCA_0.8.yaml - Higher variance
embeddings:
  image_encoder:
    pca_variance: 0.8                                    # Was 0.6 → Now 0.8

# Run:
./run_unified_pipeline.sh config/modular_Student_GEO.yaml   # Original
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.8.yaml  # Ablation
```

**Compare Results**:
- Original: `GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/training/metrics/student_training_results.json`
- Ablation: `GEO Ablation Study/STUDENT_DISTILLATION_0.8_PCA/training/metrics/student_training_results.json`

---

### Example 2: Ablate Student Architecture (Smaller Model)

```yaml
# config/modular_Student_GEO_Small.yaml
training:
  hidden_dims: [64, 32]                                  # Was [128, 64, 32] → Smaller
  learning_rate: 0.001
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_SMALL"      # Different output folder
```

**Run**:
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_Small.yaml
```

---

### Example 3: Ablate Distillation Loss Weights

```yaml
# config/modular_Student_GEO_HardLoss.yaml - Only hard loss
distillation:
  alpha: 1.0                                              # Hard loss only
  beta: 0.0                                               # No soft loss
  gamma: 0.0                                              # No feature matching
  temperature: 4.0
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_HARDLOSS"
```

**Run**:
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_HardLoss.yaml
```

---

### Example 4: Ablate Learning Rate & Optimization

```yaml
# config/modular_Student_GEO_LR_0.0001.yaml
training:
  hidden_dims: [128, 64, 32]
  learning_rate: 0.0001                                  # Lower LR
  batch_size: 16                                         # Smaller batch
  n_epochs: 200                                          # More epochs
  early_stopping_patience: 20                            # More patience
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_LR_0.0001"
```

---

## Ablation Study Workflow

### Recommended Ablation Plan

```bash
# 1. Baseline (already trained)
./run_unified_pipeline.sh config/modular_Student_GEO.yaml

# 2. PCA Ablation
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.4.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.8.yaml

# 3. Architecture Ablation
./run_unified_pipeline.sh config/modular_Student_GEO_Small.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_Large.yaml

# 4. Distillation Loss Ablation
./run_unified_pipeline.sh config/modular_Student_GEO_HardLoss.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_SoftLoss.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_FeatureLoss.yaml

# 5. Optimization Ablation
./run_unified_pipeline.sh config/modular_Student_GEO_LR_0.0001.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_LR_0.01.yaml
```

---

## Output Structure

Each pipeline run creates:

```
./GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/
├── .working/
│   ├── arrays/
│   └── preprocessed_arrays/
│       ├── fused_embeddings_pca.npy        (teacher multimodal)
│       └── uni_embeddings_pca.npy          (student morphology)
│
├── preprocessing/
│   ├── metrics/
│   └── student_pca_model.pkl               (PCA transformation)
│
├── training/
│   ├── models/
│   │   ├── fold_0_best_student_model.pth
│   │   ├── fold_1_best_student_model.pth
│   │   ├── fold_2_best_student_model.pth
│   │   ├── fold_3_best_student_model.pth
│   │   └── fold_4_best_student_model.pth
│   │
│   ├── metrics/
│   │   ├── student_training_results.json   ← ⭐ Main metrics
│   │   ├── student_predictions.csv
│   │   └── student_label_encoder.pkl
│   │
│   └── visualizations/
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       └── per_class_accuracy.png
│
└── post-training/
    ├── metrics/
    └── visualizations/
        ├── spatial_heatmaps/
        └── comparison_plots/
```

---

## Key Results

### GEO Dataset (23,342 samples)

**Teacher (Multimodal: 163D)**
- **Accuracy**: 93.96% ± 0.95%
- **Precision**: 94.15% ± 0.89%
- **Recall**: 93.96% ± 0.95%
- **F1 Score**: 94.00% ± 0.93%

**Student (Morphology-only: 18D)**
- **Accuracy**: 56.57%
- **Precision**: 54.87%
- **Recall**: 56.57%
- **F1 Score**: 55.34%

**Information Gap**: 37.39 percentage points  
→ Gene expression (scVI) + Cell composition (RCTD) = ~37% of predictive power

---

### ZENODO Dataset (16,639 samples)

**Teacher (Multimodal: 172D)**
- **Accuracy**: 85.62% ± 0.51%
- **Precision**: 86.37% ± 0.52%
- **Recall**: 85.62% ± 0.51%
- **F1 Score**: 85.82% ± 0.44%

**Student (Morphology-only: 19D)**
- **Accuracy**: 67.77%
- **Precision**: 65.22%
- **Recall**: 67.77%
- **F1 Score**: 63.48%

**Information Gap**: 17.85 percentage points  
→ Morphology is more predictive on ZENODO than GEO

---

## Understanding Results

### How to Compare Ablations

1. **Open metrics file**:
   ```bash
   cat "GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/training/metrics/student_training_results.json"
   ```

2. **Look for**:
   ```json
   {
     "overall_metrics": {
       "accuracy": 0.5657,              ← Main metric
       "precision": 0.5487,
       "recall": 0.5657,
       "f1": 0.5534
     },
     "per_fold": [
       {"fold": 1, "accuracy": 0.5620, ...},
       {"fold": 2, "accuracy": 0.5680, ...},
       ...
     ]
   }
   ```

3. **Compare across ablations**:
   - Baseline: 56.57% accuracy
   - PCA 0.4: ? accuracy
   - PCA 0.8: ? accuracy
   - Small model: ? accuracy
   - etc.

---

## Troubleshooting

### Problem: "Teacher model not found"
**Solution**: Ensure teacher was trained first or path in YAML is correct:
```yaml
distillation:
  teacher_model_path: "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA/teacher_model_FROZEN.pt"
```

### Problem: "No such file or directory" for input data
**Solution**: Check input paths in YAML match your data location:
```yaml
input_dataset:
  labels_file: "./GEO data/barcode_labels.csv"
  image_encoder_embeddings: "./GEO data/uni_embeddings_1024d_combined.csv"
```

### Problem: Out of memory
**Solution**: Reduce batch size in YAML:
```yaml
training:
  batch_size: 16  # Was 32, reduce to 16
```

### Problem: Training is slow
**Solution**: Check if GPU is available:
```bash
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')"
```

---

## Advanced: Create Custom Config Template

Copy and modify `config/modular_flexible.yaml`:

```bash
cp config/modular_flexible.yaml config/modular_Student_GEO_CUSTOM.yaml
# Edit the file with your parameters
./run_unified_pipeline.sh config/modular_Student_GEO_CUSTOM.yaml
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_unified_pipeline.sh` | Master orchestration script (one command) |
| `pipeline/train_student_model_unified.py` | Student training with distillation |
| `pipeline/build_ensemble_teacher.py` | Ensemble averaging (5 folds) |
| `config/modular_Student_GEO.yaml` | GEO student configuration |
| `config/modular_Student_Zenodo.yaml` | ZENODO student configuration |
| `config/modular_flexible.yaml` | Template for custom ablations |

---

## Next Steps

1. **Run baseline**: `./run_unified_pipeline.sh config/modular_Student_GEO.yaml`
2. **Create ablation configs**: Copy YAML files, modify parameters
3. **Run ablations**: `./run_unified_pipeline.sh config/modular_Student_GEO_VARIANT.yaml`
4. **Compare results**: Check `student_training_results.json` files
5. **Document findings**: Record accuracy/F1 changes per ablation

---

## Documentation

See `CONTEXT_DOCUMENTS/` for:
- `FINAL_STATUS_REPORT.md` - Complete pipeline validation results
- `UNIFIED_PIPELINE_TEST_REPORT.md` - Architecture details & performance analysis
- `STUDENT_MODEL_TRAINING_PLAN.md` - Original training strategy
- `FYP_GLOBAL_CONTEXT.md` - Research background & dataset info

---

## Quick Reference: Common Commands

```bash
# Train GEO student (complete pipeline)
./run_unified_pipeline.sh config/modular_Student_GEO.yaml

# Train ZENODO student (complete pipeline)
./run_unified_pipeline.sh config/modular_Zenodo.yaml

# View results
cat "GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/training/metrics/student_training_results.json"

# Compare multiple ablations
for config in config/modular_Student_GEO*.yaml; do
  echo "=== $(basename $config) ==="
  ./run_unified_pipeline.sh "$config"
done
```

---

## Support

For issues or questions:
1. Check output logs: `*_unified_pipeline.log`
2. Review error messages in terminal
3. Verify YAML syntax: `python -c "import yaml; yaml.safe_load(open('config/modular_Student_GEO.yaml'))"`
4. Check data paths exist: `ls -l ./GEO\ data/`

---

**Version**: 1.0 | **Last Updated**: March 29, 2026 | **Status**: Production Ready
