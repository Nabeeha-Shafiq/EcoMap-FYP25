# 🎯 QUICK START GUIDE - Student Model Pipeline

**GitHub Branch**: `student`  
**Status**: ✅ Ready to Use  
**Latest Commit**: Student branch fully pushed to GitHub

---

## ⚡ One-Command Training

### For GEO Dataset:
```bash
./run_unified_pipeline.sh config/modular_GEO.yaml
```

### For ZENODO Dataset:
```bash
./run_unified_pipeline.sh config/modular_Zenodo.yaml
```

**That's it!** The entire pipeline runs automatically:
1. Trains teacher (multimodal) - 16 minutes
2. Builds ensemble teacher - 1 minute  
3. Trains student (morphology-only) - 25 minutes
4. Generates all visualizations and metrics

---

## 📋 What Configuration YAML to Use

### For Different Datasets:

**GEO Dataset** (23,342 samples)
```bash
./run_unified_pipeline.sh config/modular_GEO.yaml
```
- Outputs to: `./GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/`

**ZENODO Dataset** (16,639 samples)
```bash
./run_unified_pipeline.sh config/modular_Zenodo.yaml
```
- Outputs to: `./ZENODO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/`

---

## 🔬 Ablation Studies: Quick Guide

### Method: Copy → Modify → Run

**Step 1: Copy existing config**
```bash
cp config/modular_Student_GEO.yaml config/modular_Student_GEO_VARIANT.yaml
```

**Step 2: Edit parameters**
```yaml
# Edit the new file with your changes
# E.g., change PCA variance, learning rate, architecture, etc.
```

**Step 3: Run**
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_VARIANT.yaml
```

---

## 📊 Ablation Study Examples

### Example 1: PCA Variance Ablation
**Hypothesis**: Does reducing dimensionality hurt performance?

```bash
# Create configs
cp config/modular_Student_GEO.yaml config/modular_Student_GEO_PCA_0.4.yaml
cp config/modular_Student_GEO.yaml config/modular_Student_GEO_PCA_0.8.yaml

# Edit config/modular_Student_GEO_PCA_0.4.yaml
embeddings:
  image_encoder:
    pca_variance: 0.4  # Less variance retained

# Edit config/modular_Student_GEO_PCA_0.8.yaml
embeddings:
  image_encoder:
    pca_variance: 0.8  # More variance retained

# Run all
./run_unified_pipeline.sh config/modular_Student_GEO.yaml       # Baseline 0.6
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.4.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.8.yaml

# Compare results
echo "=== BASELINE (0.6) ===" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/training/metrics/student_training_results.json" | grep -A4 overall_metrics
echo "=== PCA 0.4 ===" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.4_PCA/training/metrics/student_training_results.json" | grep -A4 overall_metrics
echo "=== PCA 0.8 ===" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.8_PCA/training/metrics/student_training_results.json" | grep -A4 overall_metrics
```

**Output comparison**:
- Baseline (0.6 PCA): ~56.57% accuracy
- PCA 0.4: ? accuracy (less information)
- PCA 0.8: ? accuracy (more information)

---

### Example 2: Student Architecture Ablation
**Hypothesis**: Does model size matter?

```yaml
# config/modular_Student_GEO_SMALL.yaml
training:
  hidden_dims: [64, 32]              # Small model
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_SMALL"

# config/modular_Student_GEO_LARGE.yaml
training:
  hidden_dims: [256, 128, 64, 32]   # Large model
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_LARGE"
```

**Run**:
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_SMALL.yaml
./run_unified_pipeline.sh config/modular_Student_GEO.yaml            # Baseline
./run_unified_pipeline.sh config/modular_Student_GEO_LARGE.yaml
```

---

### Example 3: Knowledge Distillation Loss Ablation
**Hypothesis**: Which loss component matters most?

```yaml
# config/modular_Student_GEO_HARDONLY.yaml - No knowledge distillation
distillation:
  alpha: 1.0   # Only cross-entropy
  beta: 0.0    # No soft targets
  gamma: 0.0   # No feature matching
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_HARDONLY"

# config/modular_Student_GEO_SOFT.yaml - Only soft distillation
distillation:
  alpha: 0.0   # No hard loss
  beta: 1.0    # Only soft targets
  gamma: 0.0   # No feature matching
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_SOFT"

# config/modular_Student_GEO_FEATURE.yaml - Only feature matching
distillation:
  alpha: 0.0   # No hard loss
  beta: 0.0    # No soft labels
  gamma: 1.0   # Only feature matching
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_FEATURE"
```

**Comparison**:
- No distillation (0.0 soft, 0.0 feature): Just train on labels
- Soft distillation: Does teacher's soft targets help?
- Feature matching: Do intermediate features help?
- Full distillation (baseline): All three combined

---

### Example 4: Learning Rate Ablation
**Hypothesis**: Is 0.001 optimal?

```yaml
# config/modular_Student_GEO_LR_0.0001.yaml
training:
  learning_rate: 0.0001
  batch_size: 32
  n_epochs: 200              # May need more epochs
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_LR_0.0001"

# config/modular_Student_GEO_LR_0.01.yaml
training:
  learning_rate: 0.01
  batch_size: 32
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_LR_0.01"
```

---

### Example 5: Batch Size Ablation
**Hypothesis**: Does batch size affect convergence?

```yaml
# config/modular_Student_GEO_Batch16.yaml
training:
  batch_size: 16
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_BATCH16"

# config/modular_Student_GEO_Batch64.yaml
training:
  batch_size: 64
  
output:
  output_dir: "./GEO Ablation Study/STUDENT_BATCH64"
```

---

## 📈 Complete Ablation Study Plan

### Recommended Sequence (Budget: ~7-8 hours)

**Phase 1: Baseline** (42 min)
```bash
./run_unified_pipeline.sh config/modular_Student_GEO.yaml
# ✅ Baseline: 56.57% accuracy
```

**Phase 2: Feature Engineering** (2 hours)
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.4.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_PCA_0.8.yaml
# Compare PCA variance impact
```

**Phase 3: Architecture** (2 hours)
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_SMALL.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_LARGE.yaml
# Compare model size impact
```

**Phase 4: Distillation Strategy** (2 hours)
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_HARDONLY.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_SOFT.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_FEATURE.yaml
# Compare loss component importance
```

**Phase 5: Optimization** (2 hours)
```bash
./run_unified_pipeline.sh config/modular_Student_GEO_LR_0.0001.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_LR_0.01.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_Batch16.yaml
./run_unified_pipeline.sh config/modular_Student_GEO_Batch64.yaml
# Compare learning rate and batch size
```

---

## 📊 How to Compare Results

### Extract Accuracies
```bash
# Baseline
echo "Baseline:" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/training/metrics/student_training_results.json" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"Accuracy: {d['overall_metrics']['accuracy']:.4f}, F1: {d['overall_metrics']['f1']:.4f}\")"

# PCA 0.4
echo "PCA 0.4:" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.4_PCA/training/metrics/student_training_results.json" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"Accuracy: {d['overall_metrics']['accuracy']:.4f}, F1: {d['overall_metrics']['f1']:.4f}\")"

# PCA 0.8
echo "PCA 0.8:" && cat "GEO Ablation Study/STUDENT_DISTILLATION_0.8_PCA/training/metrics/student_training_results.json" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"Accuracy: {d['overall_metrics']['accuracy']:.4f}, F1: {d['overall_metrics']['f1']:.4f}\")"
```

### Create Comparison Table
```bash
# Create CSV for Excel/analysis
for config in config/modular_Student_GEO*.yaml; do
  name=$(basename "$config" .yaml)
  result_file=$(grep "output_dir:" "$config" | cut -d: -f2 | xargs)"/training/metrics/student_training_results.json"
  if [ -f "$result_file" ]; then
    acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['overall_metrics']['accuracy'])" 2>/dev/null)
    f1=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['overall_metrics']['f1'])" 2>/dev/null)
    echo "$name,$acc,$f1"
  fi
done | tee ablation_results.csv
```

---

## 🎨 YAML Configuration Reference

### Minimum Required Changes for Ablations

```yaml
# 1. Change input data path
input_dataset:
  labels_file: "./YOUR_DATA/labels.csv"
  image_encoder_embeddings: "./YOUR_DATA/embeddings.csv"

# 2. Change output directory (IMPORTANT - prevents overwriting)
output:
  output_dir: "./YOUR_STUDY_NAME/VARIANT_NAME"

# 3. Change parameters (any/all of these)
embeddings:
  image_encoder:
    pca_variance: 0.6              # Try: 0.4, 0.5, 0.7, 0.8

training:
  hidden_dims: [128, 64, 32]       # Try: [64, 32], [256, 128, 64, 32]
  learning_rate: 0.001             # Try: 0.0001, 0.0005, 0.005, 0.01
  batch_size: 32                   # Try: 16, 64, 128
  n_epochs: 150                    # Try: 100, 200, 300
  early_stopping_patience: 15      # Try: 10, 20, 30

distillation:
  alpha: 0.7                       # Hard loss weight (try: 0, 0.5, 1.0)
  beta: 0.2                        # Soft loss weight (try: 0, 0.5, 1.0)
  gamma: 0.1                       # Feature loss weight (try: 0, 0.5, 1.0)
  temperature: 4.0                 # Softmax temp (try: 1.0, 2.0, 8.0)
```

---

## ✅ Branch Structure

**`main` branch**: Teacher training  
**`student` branch**: Student training ← You are here

All files organized:
```
student branch/
├── README.md                      ← Comprehensive guide
├── config/
│   ├── modular_GEO.yaml          (teacher)
│   ├── modular_Student_GEO.yaml  (student - baseline)
│   ├── modular_Zenodo.yaml       (teacher)
│   ├── modular_Student_Zenodo.yaml (student)
│   └── modular_flexible.yaml     (template for ablations)
├── pipeline/
│   ├── train_student_model_unified.py
│   └── build_ensemble_teacher.py
├── run_unified_pipeline.sh
└── CONTEXT_DOCUMENTS/
    ├── FINAL_STATUS_REPORT.md
    ├── UNIFIED_PIPELINE_TEST_REPORT.md
    └── ...
```

---

## 🚀 Next Steps

1. **Clone/Pull student branch**:
   ```bash
   git checkout student
   git pull
   ```

2. **Run baseline**:
   ```bash
   ./run_unified_pipeline.sh config/modular_Student_GEO.yaml
   ```

3. **Create ablation config**:
   ```bash
   cp config/modular_Student_GEO.yaml config/modular_Student_GEO_CUSTOM.yaml
   # Edit config
   ```

4. **Run ablation**:
   ```bash
   ./run_unified_pipeline.sh config/modular_Student_GEO_CUSTOM.yaml
   ```

5. **Compare results**:
   ```bash
   # Check output metrics
   cat "./YOUR_OUTPUT/training/metrics/student_training_results.json"
   ```

---

## 📞 Support

**Issues?** Check:
- ✅ All input paths in YAML are correct
- ✅ YAML syntax is valid: `python3 -c "import yaml; yaml.safe_load(open('config/modular_Student_GEO.yaml'))"`
- ✅ Data files exist: `ls -la ./GEO\ data/`
- ✅ Teacher model trained first
- ✅ Output directory path is unique (different for each ablation)

---

**Ready?** Run: `./run_unified_pipeline.sh config/modular_Student_GEO.yaml` 🎯

**GitHub**: student branch pushed ✅  
**Documentation**: README.md + CONTEXT_DOCUMENTS/  
**Status**: Production Ready 🚀
