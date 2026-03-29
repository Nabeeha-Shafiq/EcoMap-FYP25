#!/bin/bash

################################################################################
# UNIFIED TEACHER + STUDENT PIPELINE ORCHESTRATION
################################################################################
#
# Purpose:
#   Complete end-to-end knowledge distillation pipeline:
#   1. Train teacher model (multimodal) on full CV
#   2. Build ensemble teacher from all folds
#   3. Train student model (morphology-only) with distillation
#   4. Generate comprehensive metrics and visualizations
#
# Usage:
#   ./run_unified_pipeline.sh config/modular_GEO.yaml
#
#   This automatically:
#   - Trains teacher and saves results
#   - Creates ensemble teacher
#   - Trains student using ensemble teacher
#   - Generates all visualizations for both
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

echo ""
echo "================================================================================"
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     UNIFIED TEACHER + STUDENT DISTILLATION PIPELINE                  ║${NC}"
echo -e "${CYAN}║     End-to-end knowledge distillation orchestration                  ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo "================================================================================"
echo ""

if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Teacher config file required${NC}"
    echo "Usage: $0 config/modular_GEO.yaml [student_config]"
    echo ""
    echo "Examples:"
    echo "  $0 config/modular_GEO.yaml                    # Auto-select GEO student config"
    echo "  $0 config/modular_GEO.yaml config/modular_Student_GEO.yaml"
    exit 1
fi

TEACHER_CONFIG="$1"

# Auto-determine student config if not provided
if [ $# -lt 2 ]; then
    if [[ "$TEACHER_CONFIG" == *"GEO"* ]]; then
        STUDENT_CONFIG="config/modular_Student_GEO.yaml"
    elif [[ "$TEACHER_CONFIG" == *"Zenodo"* ]] || [[ "$TEACHER_CONFIG" == *"ZENODO"* ]]; then
        STUDENT_CONFIG="config/modular_Student_Zenodo.yaml"
    else
        echo -e "${RED}Error: Could not auto-determine student config${NC}"
        echo "Please specify: $0 <teacher_config> <student_config>"
        exit 1
    fi
else
    STUDENT_CONFIG="$2"
fi

if [ ! -f "$TEACHER_CONFIG" ]; then
    echo -e "${RED}Error: Teacher config not found: $TEACHER_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$STUDENT_CONFIG" ]; then
    echo -e "${RED}Error: Student config not found: $STUDENT_CONFIG${NC}"
    exit 1
fi

echo -e "Teacher config: ${BLUE}$TEACHER_CONFIG${NC}"
echo -e "Student config: ${BLUE}$STUDENT_CONFIG${NC}"
echo ""

# Get teacher output directory
TEACHER_OUTPUT_DIR=$($PYTHON_EXEC -c "
import yaml
with open('$TEACHER_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('output', {}).get('output_dir', './results'))
" 2>/dev/null)

# Get student output directory
STUDENT_OUTPUT_DIR=$($PYTHON_EXEC -c "
import yaml
with open('$STUDENT_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('output', {}).get('output_dir', './results'))
" 2>/dev/null)

echo -e "Teacher output: ${GREEN}$TEACHER_OUTPUT_DIR${NC}"
echo -e "Student output: ${GREEN}$STUDENT_OUTPUT_DIR${NC}"
echo ""

# Create all necessary directories upfront
echo -e "${CYAN}Setting up output directories...${NC}"
mkdir -p "$TEACHER_OUTPUT_DIR"/{preprocessing,training/{models,metrics,visualizations},post-training/{metrics,visualizations}}
mkdir -p "$TEACHER_OUTPUT_DIR"/.working/{arrays,preprocessed_arrays}
mkdir -p "$STUDENT_OUTPUT_DIR"/{preprocessing,training/{models,metrics,visualizations},post-training/{metrics,visualizations}}
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

################################################################################
# PHASE 1: TRAIN TEACHER MODEL
################################################################################

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PHASE 1/3: TRAINING TEACHER MODEL (Multimodal)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

chmod +x run_pipeline.sh
./run_pipeline.sh "$TEACHER_CONFIG"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Teacher training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Teacher training complete${NC}"
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
    echo -e "${RED}✗ Ensemble teacher building failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ensemble teacher ready${NC}"
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
    echo -e "${RED}✗ Student training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Student training complete${NC}"
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
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Review teacher metrics: $TEACHER_OUTPUT_DIR/training/metrics/"
echo "  2. Review student metrics: $STUDENT_OUTPUT_DIR/training/metrics/"
echo "  3. Compare visualizations: Check post-training/ subdirectories"
echo ""
echo "================================================================================"

exit 0
