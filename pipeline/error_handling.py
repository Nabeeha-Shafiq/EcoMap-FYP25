"""
================================================================================
ERROR HANDLING & VALIDATION UTILITIES
================================================================================

Purpose:
    Early detection and clear error messages for common issues in the 
    unified teacher-student pipeline.

Philosophy:
    - Fail fast with clear messages
    - Point users to exact problem and solution
    - Prevent silent failures with dimension mismatches
    - Validate configuration before training begins

Available Checks:
    1. Embedding File Validation
    2. Dimension Consistency Check
    3. PCA Model Persistence
    4. Configuration Validation
    5. Path Existence Verification
    6. Embedding File Format Validation

Usage:
    from pipeline.error_handling import ConfigValidator, DimensionChecker
    
    # Early in pipeline:
    validator = ConfigValidator(config_file)
    validator.validate_all()  # Raises exceptions with helpful messages
    
    # Before training:
    dim_checker = DimensionChecker(teacher_config, student_config)
    dim_checker.verify_consistency()  # Catches embedding mismatches

================================================================================
"""

import os
import sys
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class PipelineError(Exception):
    """Base exception for pipeline errors with user-friendly messages"""
    pass


class EmbeddingMismatchError(PipelineError):
    """Embedding files between teacher and student don't match"""
    pass


class DimensionMismatchError(PipelineError):
    """Output dimensions don't match expected values"""
    pass


class MissingFileError(PipelineError):
    """Required file not found"""
    pass


class ConfigurationError(PipelineError):
    """Configuration validation failed"""
    pass


class PCAModelError(PipelineError):
    """PCA model loading or saving failed"""
    pass


# ============================================================================
# COLOR OUTPUT FOR TERMINAL
# ============================================================================

class Color:
    """Terminal color codes for better error visibility"""
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    GREEN = '\033[0;32m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_error(title: str, message: str, solution: Optional[str] = None):
    """Print formatted error message"""
    print(f"\n{Color.RED}{'=' * 80}{Color.RESET}")
    print(f"{Color.RED}{Color.BOLD}✗ ERROR: {title}{Color.RESET}")
    print(f"{Color.RED}{'=' * 80}{Color.RESET}")
    print(f"\n  {message}\n")
    
    if solution:
        print(f"{Color.CYAN}{Color.BOLD}💡 SOLUTION:{Color.RESET}")
        for line in solution.split('\n'):
            print(f"  {line}")
    
    print(f"\n{Color.RED}{'=' * 80}{Color.RESET}\n")


def print_warning(title: str, message: str):
    """Print formatted warning message"""
    print(f"\n{Color.YELLOW}{Color.BOLD}⚠ WARNING: {title}{Color.RESET}")
    print(f"  {message}\n")


def print_success(title: str, message: str):
    """Print formatted success message"""
    print(f"{Color.GREEN}{'✓'} {title}{Color.RESET}")
    if message:
        print(f"  {message}")


# ============================================================================
# FILE & PATH VALIDATION
# ============================================================================

class FileValidator:
    """Validates existence and format of input files"""
    
    @staticmethod
    def check_file_exists(file_path: str, file_type: str = "file") -> bool:
        """Check if file exists, with helpful error message"""
        if not os.path.exists(file_path):
            solution = (
                f"1. Verify the file path is correct:\n"
                f"     Expected: {file_path}\n"
                f"2. Check if the file was downloaded\n"
                f"3. Ensure you're in the correct working directory:\n"
                f"     pwd: {os.getcwd()}\n"
                f"     file: {Path(file_path).resolve()}"
            )
            raise MissingFileError(
                f"{file_type} not found: {file_path}"
            )
        return True
    
    @staticmethod
    def check_csv_readable(csv_file: str, min_rows: int = 10) -> Tuple[int, int]:
        """
        Check if CSV file is readable and returns (n_rows, n_cols)
        
        Args:
            csv_file: Path to CSV file
            min_rows: Minimum rows expected
            
        Returns:
            Tuple of (n_rows, n_cols)
            
        Raises:
            MissingFileError: If file doesn't exist
            ConfigurationError: If file format is invalid
        """
        FileValidator.check_file_exists(csv_file, "CSV file")
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 2:  # At least header + 1 row
                raise ConfigurationError(
                    f"CSV file too small: {csv_file}\n"
                    f"Expected at least 2 rows (header + data), got {len(rows)}\n"
                    f"\nSolution:\n"
                    f"- Verify the CSV file is not corrupted\n"
                    f"- Check file size: {os.path.getsize(csv_file)} bytes"
                )
            
            n_rows = len(rows) - 1  # Exclude header
            n_cols = len(rows[0])
            
            return n_rows, n_cols
            
        except Exception as e:
            solution = (
                f"1. Verify CSV file is readable:\n"
                f"     file: {csv_file}\n"
                f"2. Check file encoding (should be UTF-8)\n"
                f"3. Verify CSV format:\n"
                f"     head -5 {csv_file}"
            )
            raise ConfigurationError(
                f"Cannot read CSV file: {csv_file}\n"
                f"Error: {str(e)}"
            )
    
    @staticmethod
    def get_csv_dimensions(csv_file: str) -> int:
        """Get number of columns (embedding dimensions) in CSV"""
        _, n_cols = FileValidator.check_csv_readable(csv_file)
        return n_cols


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

class ConfigValidator:
    """Validates unified configuration files"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load and parse YAML configuration"""
        FileValidator.check_file_exists(self.config_file, "Config file")
        
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            solution = (
                f"1. Verify YAML syntax is correct\n"
                f"2. Check indentation (spaces, not tabs)\n"
                f"3. Use online YAML validator: yamlint.com"
            )
            raise ConfigurationError(
                f"Cannot parse YAML config: {self.config_file}\n"
                f"Error: {str(e)}"
            )
    
    def validate_all(self):
        """Run all validation checks"""
        print_success("Config Validation", "Starting...")
        
        self.check_required_sections()
        self.check_required_fields()
        self.check_input_files()
        self.check_output_directories()
        
        print_success("All Validations", "PASSED ✓")
    
    def check_required_sections(self):
        """Verify required YAML sections exist"""
        required = ['dataset', 'input_dataset', 'embeddings', 'teacher', 'student']
        
        for section in required:
            if section not in self.config:
                solution = (
                    f"The config file is missing the '{section}' section.\n"
                    f"Use one of the template configs:\n"
                    f"  - config/modular_unified_teacher_student_GEO.yaml\n"
                    f"  - config/modular_unified_teacher_student_Zenodo.yaml"
                )
                raise ConfigurationError(
                    f"Missing required section: '{section}' in {self.config_file}"
                )
        
        print_success("Required Sections", "OK")
    
    def check_required_fields(self):
        """Verify required fields in each section"""
        requirements = {
            'input_dataset': ['labels_file', 'image_encoder_embeddings'],
            'teacher': ['model_type', 'output_dir'],
            'student': ['model_type', 'output_dir'],
        }
        
        for section, fields in requirements.items():
            if section not in self.config:
                continue
            
            config_section = self.config[section]
            for field in fields:
                if field not in config_section or not config_section[field]:
                    raise ConfigurationError(
                        f"Missing required field '{field}' in [{section}] section"
                    )
        
        print_success("Required Fields", "OK")
    
    def check_input_files(self):
        """Verify all input embedding files exist"""
        embeddings = self.config.get('embeddings', {})
        
        for emb_name, emb_config in embeddings.items():
            if isinstance(emb_config, dict) and 'file_path' in emb_config:
                file_path = emb_config['file_path']
                try:
                    FileValidator.check_file_exists(file_path, f"{emb_name}")
                except MissingFileError:
                    raise ConfigurationError(
                        f"Embedding file not found for {emb_name}: {file_path}\n"
                        f"Available files:\n"
                        f"  ls -la GEO\\ data/input_dataset/\n"
                        f"  or\n"
                        f"  ls -la ZENODO\\ data/"
                    )
        
        print_success("Input Files", "All present")
    
    def check_output_directories(self):
        """Verify output directories can be created"""
        teacher_out = self.config.get('teacher', {}).get('output_dir')
        student_out = self.config.get('student', {}).get('output_dir')
        
        for out_dir, role in [(teacher_out, 'teacher'), (student_out, 'student')]:
            if out_dir:
                parent_dir = str(Path(out_dir).parent)
                if not os.path.exists(parent_dir):
                    try:
                        os.makedirs(parent_dir, exist_ok=True)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Cannot create output directory: {parent_dir}\n"
                            f"Error: {str(e)}"
                        )
        
        print_success("Output Directories", "OK")


# ============================================================================
# DIMENSION CONSISTENCY CHECKING
# ============================================================================

class DimensionChecker:
    """Verifies embedding dimensions are consistent between teacher and student"""
    
    def __init__(self, teacher_config: Dict[str, Any], student_config: Dict[str, Any]):
        self.teacher_config = teacher_config
        self.student_config = student_config
    
    def verify_consistency(self):
        """Check that teacher and student use same embedding files"""
        print_success("Dimension Validation", "Starting...")
        
        # Get embedding file paths
        teacher_embeddings = self.teacher_config.get('input_dataset', {})
        student_embeddings = self.student_config.get('input_dataset', {})
        
        # Check image encoder (most critical)
        self._check_embedding_match(
            teacher_embeddings.get('image_encoder_embeddings'),
            student_embeddings.get('image_encoder_embeddings'),
            'image_encoder_embeddings'
        )
        
        # Check gene encoder
        self._check_embedding_match(
            teacher_embeddings.get('gene_embeddings'),
            student_embeddings.get('gene_embeddings'),
            'gene_embeddings'
        )
        
        print_success("Dimension Consistency", "OK - Embeddings match")
    
    def _check_embedding_match(self, teacher_file: str, student_file: str, 
                               embedding_name: str):
        """Check specific embedding file matches"""
        if teacher_file != student_file:
            solution = (
                f"Both teacher and student must use the same embedding files.\n"
                f"\nCurrent mismatch:\n"
                f"  Teacher: {teacher_file}\n"
                f"  Student: {student_file}\n"
                f"\nFix:\n"
                f"  Edit your unified config file and ensure both sections use:\n"
                f"  input_dataset:\n"
                f"    {embedding_name}: {teacher_file}"
            )
            raise EmbeddingMismatchError(
                f"Embedding file mismatch for {embedding_name}"
            )
    
    def check_pca_settings(self):
        """Verify PCA variance settings match"""
        teacher_pca = self.teacher_config.get('embeddings', {}).get(
            'image_encoder', {}
        ).get('pca_variance')
        
        student_pca = self.student_config.get('embeddings', {}).get(
            'image_encoder', {}
        ).get('pca_variance')
        
        if teacher_pca != student_pca:
            print_warning(
                "PCA Variance Mismatch",
                f"Teacher: {teacher_pca}, Student: {student_pca}\n"
                f"This may lead to dimension mismatches.\n"
                f"Recommendation: Use same PCA variance for both"
            )


# ============================================================================
# PCA MODEL VALIDATION
# ============================================================================

class PCAValidator:
    """Validates PCA models are properly saved and accessible"""
    
    def __init__(self, teacher_output_dir: str):
        self.teacher_output_dir = teacher_output_dir
        self.pca_dir = os.path.join(
            teacher_output_dir,
            '.working/preprocessed_arrays/pca_models'
        )
    
    def verify_models_exist(self) -> bool:
        """Check if PCA models were saved by teacher"""
        if not os.path.exists(self.pca_dir):
            print_warning(
                "PCA Models Not Found",
                f"Expected directory: {self.pca_dir}\n"
                f"The student will fit new PCA models if needed.\n"
                f"However, consistency is better if teacher PCA is reused."
            )
            return False
        
        # Count PCA model files
        pca_files = [f for f in os.listdir(self.pca_dir) if f.endswith('.pkl')]
        
        if not pca_files:
            print_warning(
                "PCA Model Files Not Found",
                f"Directory exists but no .pkl files found: {self.pca_dir}"
            )
            return False
        
        print_success("PCA Models", f"Found {len(pca_files)} models")
        return True
    
    def verify_accessibility(self) -> bool:
        """Check if PCA models are readable"""
        if not self.verify_models_exist():
            return False
        
        try:
            pca_files = [f for f in os.listdir(self.pca_dir) if f.endswith('.pkl')]
            for pca_file in pca_files:
                file_path = os.path.join(self.pca_dir, pca_file)
                # Try to read first 100 bytes
                with open(file_path, 'rb') as f:
                    _ = f.read(100)
            
            print_success("PCA Model Accessibility", "All files readable")
            return True
            
        except Exception as e:
            raise PCAModelError(
                f"Cannot read PCA model files: {str(e)}\n"
                f"Check file permissions: chmod +r {self.pca_dir}/*"
            )


# ============================================================================
# COMPREHENSIVE VALIDATION PIPELINE
# ============================================================================

class PipelineValidator:
    """Comprehensive validation for entire pipeline before training"""
    
    def __init__(self, unified_config_file: str):
        self.config_file = unified_config_file
        self.validator = ConfigValidator(config_file)
    
    def validate_before_training(self):
        """Run all checks before training starts"""
        print(f"\n{'=' * 80}")
        print(f"{Color.BOLD}COMPREHENSIVE PIPELINE VALIDATION{Color.RESET}")
        print(f"{'=' * 80}\n")
        
        try:
            # Step 1: Validate configuration
            self.validator.validate_all()
            
            # Step 2: Check dimension consistency
            dim_checker = DimensionChecker(
                self.validator.config.get('teacher', {}),
                self.validator.config.get('student', {})
            )
            dim_checker.verify_consistency()
            
            # Step 3: Check PCA models if teacher already ran
            teacher_output = self.validator.config.get('teacher', {}).get('output_dir')
            if teacher_output and os.path.exists(teacher_output):
                pca_validator = PCAValidator(teacher_output)
                pca_validator.verify_accessibility()
            
            print(f"\n{'=' * 80}")
            print(f"{Color.GREEN}{Color.BOLD}✓ ALL VALIDATION CHECKS PASSED{Color.RESET}")
            print(f"{'=' * 80}\n")
            
            return True
            
        except PipelineError as e:
            print_error(e.__class__.__name__, str(e))
            sys.exit(1)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR DIRECT USE
# ============================================================================

def validate_config(config_file: str):
    """Quick config validation"""
    validator = ConfigValidator(config_file)
    validator.validate_all()


def check_embedding_consistency(config1: Dict, config2: Dict):
    """Check two configs have consistent embeddings"""
    checker = DimensionChecker(config1, config2)
    checker.verify_consistency()


def validate_before_training(unified_config_file: str):
    """Full validation before training"""
    validator = PipelineValidator(unified_config_file)
    validator.validate_before_training()


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python error_handling.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    validate_before_training(config_file)
