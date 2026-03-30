#!/bin/bash

################################################################################
# UNIFIED TEACHER + STUDENT PIPELINE ORCHESTRATION
################################################################################
#
# Purpose:
#   Complete end-to-end knowledge distillation pipeline from ONE config file:
#   1. Extract teacher and student configs from unified config
#   2. Train teacher model (multimodal) on full CV
#   3. Build ensemble teacher from all folds
#   4. Train student model (morphology-only) with distillation
#   5. Generate comprehensive metrics and visualizations for both
#
# Usage:
#   ./run_unified_pipeline.sh config/modular_unified_teacher_student_GEO.yaml
#
################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate venv
source venv/bin/activate
PYTHON_EXEC="./venv/bin/python"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

echo ""
echo "================================================================================"
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     UNIFIED TEACHER + STUDENT DISTILLATION PIPELINE                  ║${NC}"
echo -e "${CYAN}║     Single Configuration File • End-to-End Orchestration             ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo "================================================================================"
echo ""

# Validate input
if [ $# -lt 1 ]; then
    log_error "Unified config file required"
    echo "Usage: $0 config/modular_unified_teacher_student_GEO.yaml"
    exit 1
fi

UNIFIED_CONFIG="$1"

if [ ! -f "$UNIFIED_CONFIG" ]; then
    log_error "Config not found: $UNIFIED_CONFIG"
    exit 1
fi

log_success "Unified config found: $UNIFIED_CONFIG"
echo ""

# Extract teacher and student configs
log_info "Extracting teacher and student configs..."
CONFIGS=$($PYTHON_EXEC pipeline/extract_configs.py "$UNIFIED_CONFIG")
TEACHER_CONFIG=$(echo "$CONFIGS" | sed -n '1p')
STUDENT_CONFIG=$(echo "$CONFIGS" | sed -n '2p')
TEACHER_OUTPUT_DIR=$(echo "$CONFIGS" | sed -n '3p')

if [ ! -f "$TEACHER_CONFIG" ]; then
    log_error "Failed to create teacher config"
    exit 1
fi

if [ ! -f "$STUDENT_CONFIG" ]; then
    log_error "Failed to create student config"
    exit 1
fi

# Extract student output directory from student config
STUDENT_OUTPUT_DIR=$($PYTHON_EXEC -c "
import yaml
with open('$STUDENT_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('output', {}).get('output_dir', './results'))
" 2>/dev/null)

log_success "Teacher config: $TEACHER_CONFIG"
log_success "Student config: $STUDENT_CONFIG"
log_success "Teacher output: $TEACHER_OUTPUT_DIR"
log_success "Student output: $STUDENT_OUTPUT_DIR"
echo ""

# Create all necessary directories upfront
echo -e "${CYAN}Setting up output directories...${NC}"
mkdir -p "$TEACHER_OUTPUT_DIR"/{preprocessing,training/{models,metrics,visualizations},post-training/{metrics,visualizations}}
mkdir -p "$TEACHER_OUTPUT_DIR"/.working/{arrays,preprocessed_arrays}
mkdir -p "$STUDENT_OUTPUT_DIR"/{preprocessing,training/{models,metrics,visualizations},post-training/{metrics,visualizations}}
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

################################################################################
# STAGE 0: Load Input Embeddings
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STAGE 0/3: Loading Input Embeddings${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON_EXEC pipeline/load_input_embeddings.py \
    --config "$TEACHER_CONFIG" \
    --output-dir "$TEACHER_OUTPUT_DIR/.working/arrays"

if [ ! -f "$TEACHER_OUTPUT_DIR/.working/arrays/barcodes.npy" ]; then
    log_error "Embedding loading failed - no output files created"
    rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"
    exit 1
fi

log_success "Input embeddings loaded"
echo ""

################################################################################
# STAGE 0.5: Preprocessing (PCA Reduction)
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STAGE 0.5/3: Applying PCA Preprocessing${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON_EXEC pipeline/preprocess_embeddings.py \
    --config "$TEACHER_CONFIG" \
    --input-arrays-dir "$TEACHER_OUTPUT_DIR/.working/arrays" \
    --output-dir "$TEACHER_OUTPUT_DIR/.working/preprocessed_arrays"

if [ ! -f "$TEACHER_OUTPUT_DIR/.working/preprocessed_arrays/fused_embeddings_pca.npy" ]; then
    log_error "Preprocessing failed - fused embeddings not created"
    rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"
    exit 1
fi

log_success "PCA preprocessing complete"
cp "$TEACHER_OUTPUT_DIR/.working/preprocessed_arrays/preprocessing_report.yaml" "$TEACHER_OUTPUT_DIR/preprocessing/" 2>/dev/null || true
echo ""

################################################################################
# PHASE 1: TRAIN TEACHER MODEL
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PHASE 1/3: TRAINING TEACHER MODEL (Multimodal)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON_EXEC pipeline/train_mlp.py --config "$TEACHER_CONFIG"

if [ $? -ne 0 ]; then
    log_error "Teacher training failed"
    rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"
    exit 1
fi

log_success "Teacher training complete"
echo ""

################################################################################
# PHASE 2: BUILD ENSEMBLE TEACHER
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PHASE 2/3: BUILDING ENSEMBLE TEACHER (Averaging all 5 folds)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON_EXEC pipeline/build_ensemble_teacher.py --teacher_output "$TEACHER_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    log_error "Ensemble teacher building failed"
    rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"
    exit 1
fi

log_success "Ensemble teacher ready"
echo ""

################################################################################
# PHASE 3: TRAIN STUDENT MODEL
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PHASE 3/3: TRAINING STUDENT MODEL (Morphology-only with Distillation)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON_EXEC pipeline/train_student_model_unified.py --config "$STUDENT_CONFIG"

if [ $? -ne 0 ]; then
    log_error "Student training failed"
    rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"
    exit 1
fi

log_success "Student training complete"
echo ""

################################################################################
# COMPLETION
################################################################################

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ COMPLETE PIPELINE EXECUTION SUCCESSFUL${NC}"
echo "================================================================================"
echo ""
echo -e "${GREEN}Teacher Results:${NC}"
echo -e "  Output: ${BLUE}$TEACHER_OUTPUT_DIR${NC}"
echo ""
echo -e "${GREEN}Student Results:${NC}"
echo -e "  Output: ${BLUE}$STUDENT_OUTPUT_DIR${NC}"
echo ""
echo -e "${CYAN}Review Results:${NC}"
echo "  Teacher metrics: $TEACHER_OUTPUT_DIR/training/metrics/"
echo "  Student metrics: $STUDENT_OUTPUT_DIR/training/metrics/"
echo "  Visualizations: $TEACHER_OUTPUT_DIR/post-training/ and $STUDENT_OUTPUT_DIR/post-training/"
echo ""
echo "================================================================================"

# Cleanup temp configs
rm -f "$TEACHER_CONFIG" "$STUDENT_CONFIG"

exit 0
