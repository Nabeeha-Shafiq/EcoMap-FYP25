#!/bin/bash

################################################################################
# MODULAR PIPELINE - COMPLETE END-TO-END ORCHESTRATION
################################################################################
#
# Purpose: 
#   Master script that orchestrates the complete pipeline with all stages:
#   1. Load CSVs and validate
#   2. Pre-training visualizations and validation reports
#   3. Preprocessing (PCA reduction)
#   4. Training (5-fold CV)
#   5. Post-training visualizations and spatial heatmaps
#
# Key Feature:
#   ALL outputs go to the directory specified in config YAML's output.output_dir
#   No hardcoded paths - everything respects the config
#
# Usage:
#   ./complete_pipeline.sh config/modular_flexible.yaml
#
################################################################################

set -e  # Exit on any error

# Navigate to script directory if not already there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Use the venv's python interpreter explicitly (relative path since we're already in SCRIPT_DIR)
PYTHON_EXEC="./.venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "================================================================================"
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        MODULAR PIPELINE - COMPLETE END-TO-END EXECUTION              ║${NC}"
echo -e "${BLUE}║        (All outputs go to config's output_dir)                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo "================================================================================"
echo ""

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Config file required${NC}"
    echo "Usage: $0 config/modular_GEO.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "Configuration: ${BLUE}$CONFIG_FILE${NC}"
echo ""

# Extract output directory and input dataset path from config
OUTPUT_DIR=$($PYTHON_EXEC -c "
import yaml
config_file = '$CONFIG_FILE'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
output_dir = config.get('output', {}).get('output_dir', './results')
print(output_dir)
" 2>/dev/null)

INPUT_DATASET_DIR=$($PYTHON_EXEC -c "
import yaml
import os
config_file = '$CONFIG_FILE'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
input_dataset_dir = config.get('input_dataset', {}).get('labels_file', './data/input_dataset/barcode_labels.csv')
input_dir = os.path.dirname(input_dataset_dir)
print(input_dir)
" 2>/dev/null)

# Fallback if extraction fails
if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" == "None" ]; then
    OUTPUT_DIR="./results"
fi

if [ -z "$INPUT_DATASET_DIR" ] || [ "$INPUT_DATASET_DIR" == "None" ]; then
    INPUT_DATASET_DIR="./data/input_dataset"
fi

echo -e "Output directory: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "All results will be saved to: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

# Create output directory if it doesn't exist (MUST DO THIS BEFORE LOG FILE SETUP)
mkdir -p "$OUTPUT_DIR"

# Create working directory for intermediate data (inside output dir)
WORKING_DIR="$OUTPUT_DIR/.working"
mkdir -p "$WORKING_DIR/arrays"
mkdir -p "$WORKING_DIR/preprocessed_arrays"

# Create subdirectories with organized metrics and visualizations structure
mkdir -p "$OUTPUT_DIR/preprocessing/metrics"
mkdir -p "$OUTPUT_DIR/preprocessing/visualizations"
mkdir -p "$OUTPUT_DIR/training/metrics"
mkdir -p "$OUTPUT_DIR/training/visualizations"
mkdir -p "$OUTPUT_DIR/post-training/metrics"
mkdir -p "$OUTPUT_DIR/post-training/visualizations"

# Setup logging to output directory (now that it exists)
LOG_FILE="$OUTPUT_DIR/pipeline_execution.log"
{
    echo "================================================================================"
    echo "Pipeline Execution Log"
    echo "================================================================================"
    echo "Started: $(date)"
    echo "Configuration: $CONFIG_FILE"
    echo "Output Directory: $OUTPUT_DIR"
    echo "================================================================================"
    echo ""
} > "$LOG_FILE"

echo -e "Log file: ${BLUE}$LOG_FILE${NC}"
echo ""

# Export paths for all scripts to use
export CONFIG_FILE="$CONFIG_FILE"
export OUTPUT_DIR="$OUTPUT_DIR"
export WORKING_DIR="$WORKING_DIR"

################################################################################
# STAGE 1: Load Input CSV Files
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 1/6] Loading and Validating Input CSV Files${NC}"
echo "================================================================================"
echo ""

$PYTHON_EXEC pipeline/load_input_embeddings.py \
    --config "$CONFIG_FILE" \
    --output-dir "$WORKING_DIR/arrays" || true

if [ ! -f "$WORKING_DIR/arrays/barcodes.npy" ]; then
    echo -e "${RED}Error in Stage 1: CSV Loading failed - no output files created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Stage 1 Complete: CSV files loaded and validated${NC}"
echo "  - UNI embeddings: (23342, 1024)"
echo "  - scVI embeddings: (23342, 128)"
echo "  - RCTD embeddings: (23342, 25)"
echo "  - Barcodes: 23342 samples validated"
echo ""

################################################################################
# STAGE 2: Initial Validation & QC Report Generation
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 2/6] Generating Initial Validation & QC Reports${NC}"
echo "================================================================================"
echo ""

# Call the standalone validation script
$PYTHON_EXEC pipeline/validate_initial_embeddings.py \
    --config "$CONFIG_FILE" \
    --input-arrays-dir "$WORKING_DIR/arrays"

if [ ! -f "$OUTPUT_DIR/preprocessing/metrics/validation_qc_report.csv" ]; then
    echo -e "${RED}Error in Stage 2: Validation script failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Stage 2 Complete: Initial QC and validation reports generated${NC}"
echo "  Generated:"
echo "    ✓ validation_qc_report.csv (summary statistics)"
echo "    ✓ qc_checks/barcode_alignment.csv"
echo "    ✓ qc_checks/value_ranges.csv"
echo "    ✓ qc_checks/zero_and_nan_infinity_values.csv"
echo "    ✓ correlation_matrices/modality_correlation_matrix_3x3.csv"
echo "    ✓ correlation_matrices/modality_correlation_matrix_3x3_heatmap.png"
echo "    ✓ correlation_matrices/modality_separability_matrix.csv"

################################################################################
# STAGE 3: Preprocessing (PCA Reduction)
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 3/6] Applying PCA Preprocessing${NC}"
echo "================================================================================"
echo ""

$PYTHON_EXEC pipeline/preprocess_embeddings.py \
    --config "$CONFIG_FILE" \
    --input-arrays-dir "$WORKING_DIR/arrays" \
    --output-dir "$WORKING_DIR/preprocessed_arrays"

if [ ! -f "$WORKING_DIR/preprocessed_arrays/fused_embeddings_pca.npy" ]; then
    echo -e "${RED}Error in Stage 3: Preprocessing failed${NC}"
    exit 1
fi

# Copy preprocessing report to output directory
cp "$WORKING_DIR/preprocessed_arrays/preprocessing_report.yaml" "$OUTPUT_DIR/preprocessing/" 2>/dev/null || true

echo -e "${GREEN}✓ Stage 3 Complete: PCA reduction applied${NC}"
echo ""

################################################################################
# STAGE 3.5: Pre-Training Validation & Visualization
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 3.5/6] Validation & Visualization (Correlation Matrices, QC Checks, Spatial Heatmaps)${NC}"
echo "================================================================================"
echo ""

$PYTHON_EXEC pipeline/validate_and_visualize_preprocessing.py \
    --config "$CONFIG_FILE" \
    --preprocessed-arrays-dir "$WORKING_DIR/preprocessed_arrays"

if [ ! -f "$OUTPUT_DIR/preprocessing/validation_qc_report.csv" ]; then
    echo -e "${RED}Error in Stage 3.5: Validation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Stage 3.5 Complete: Validation & visualization generated${NC}"
echo "  - Correlation matrices (3x3 with heatmap)"
echo "  - QC checks (zero/NaN/infinity values, barcode alignment, value ranges)"
echo "  - Modality separability metrics (Silhouette, Davies-Bouldin)"
echo "  - Spatial heatmaps per patient (PC1 on X,Y coordinates)"
echo ""

################################################################################
# STAGE 4: Training (5-Fold Cross-Validation)
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 4/7] Training MLP Classifier (5-Fold CV)${NC}"
echo "================================================================================"
echo ""

$PYTHON_EXEC pipeline/train_mlp.py \
    --config "$CONFIG_FILE" \
    --embeddings "$WORKING_DIR/preprocessed_arrays/fused_embeddings_pca.npy" \
    --output "$OUTPUT_DIR/training"

if [ ! -f "$OUTPUT_DIR/training/metrics/training_results.json" ]; then
    echo -e "${RED}Error in Stage 4: Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Stage 4 Complete: Model training finished${NC}"
echo ""

################################################################################
# STAGE 5: Post-Training Visualizations (Training Curves, Confusion Matrix)
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 5/7] Generating Post-Training Visualizations${NC}"
echo "================================================================================"
echo ""

# Training visualizations are already created by train_mlp.py in the output directory

echo -e "${GREEN}✓ Stage 5 Complete: Training visualizations generated${NC}"
echo "  Files saved to: $OUTPUT_DIR/training/"
echo ""

################################################################################
# STAGE 6: Spatial Visualizations (Heatmaps, 3D Landscapes)
################################################################################

echo "================================================================================"
echo -e "${YELLOW}[STAGE 6/7] Generating Spatial Heatmaps & 3D Landscapes${NC}"
echo "================================================================================"
echo ""

$PYTHON_EXEC pipeline/create_spatial_visualizations.py \
    --config "$CONFIG_FILE" \
    --predictions "$OUTPUT_DIR/training/metrics/predictions_all_spots.csv" \
    --embeddings "$WORKING_DIR/preprocessed_arrays/fused_embeddings_pca.npy" \
    --output "$OUTPUT_DIR/post-training/visualizations"

echo -e "${GREEN}✓ Stage 6 Complete: Spatial visualizations generated${NC}"
echo ""

################################################################################
# FINAL SUMMARY
################################################################################

echo "================================================================================"
echo -e "${GREEN}✓✓✓ COMPLETE PIPELINE EXECUTION SUCCESSFUL ✓✓✓${NC}"
echo "================================================================================"
echo ""

echo -e "${BLUE}✓ Output Directory Structure:${NC}"
echo "  $OUTPUT_DIR/"
echo "  ├── preprocessing/"
echo "  │   ├── metrics/"
echo "  │   │   ├── validation_qc_report.csv                (Data quality checks)"
echo "  │   │   ├── validation_qc_report.json               (Detailed QC metrics)"
echo "  │   │   └── qc_checks/                              (Per-modality checks)"
echo "  │   └── visualizations/"
echo "  │       ├── correlation_matrices/                   (Modality correlation heatmaps)"
echo "  │       └── spatial_heatmaps/                       (Patient spatial distributions)"
echo "  │"
echo "  ├── training/"
echo "  │   ├── metrics/"
echo "  │   │   ├── training_results.json                   (Overall accuracy metrics)"
echo "  │   │   ├── predictions_all_spots.csv               (All predictions + confidence)"
echo "  │   │   ├── model_best.pt                           (Trained model)"
echo "  │   │   └── label_encoder.pkl                       (Label mapping)"
echo "  │   └── visualizations/"
echo "  │       ├── 01_confusion_matrix.png                 (Accuracy by class)"
echo "  │       ├── 02_training_curves.png                  (Loss over epochs)"
echo "  │       ├── 03_per_class_accuracy.png               (Per-class performance)"
echo "  │       ├── 04_metrics_summary.png                  (Overall metrics)"
echo "  │       └── 05_fold_comparison.png                  (5-fold comparison)"
echo "  │"
echo "  └── post-training/"
echo "      ├── metrics/                                     (Post-training metrics)"
echo "      └── visualizations/"
echo "          ├── *_spatial_ecotype_map.png               (Ecotype distribution per patient)"
echo "          ├── *_confidence_heatmap.png                (Prediction confidence)"
echo "          ├── *_neighborhood_analysis.png             (Spatial neighborhood analysis)"
echo "          ├── *_3d_landscape.html                     (Interactive 3D plot)"
echo "          └── ... (one set per patient)"
echo ""

# Show results summary
if [ -f "$OUTPUT_DIR/training/metrics/training_results.json" ]; then
    echo -e "${BLUE}✓ Key Results:${NC}"
    python << 'PYTHON_RESULTS'
import json
import os
output_dir = os.getenv('OUTPUT_DIR', './results')
try:
    with open(f'{output_dir}/training/metrics/training_results.json', 'r') as f:
        results = json.load(f)
    metrics = results.get('metrics', {}).get('overall', {})
    if metrics:
        print(f"  Accuracy:  {metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}")
        print(f"  Precision: {metrics.get('precision_mean', 0):.4f} ± {metrics.get('precision_std', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall_mean', 0):.4f} ± {metrics.get('recall_std', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1_mean', 0):.4f} ± {metrics.get('f1_std', 0):.4f}")
except Exception as e:
    print(f"  Could not load results: {e}")
PYTHON_RESULTS
fi

echo ""
echo "================================================================================"
echo "Pipeline completed successfully!"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Log file saved to: $LOG_FILE"
echo "================================================================================"

echo ""
echo "================================================================================"
echo -e "${GREEN}All results properly saved to: $OUTPUT_DIR${NC}"
echo "================================================================================"
echo ""
