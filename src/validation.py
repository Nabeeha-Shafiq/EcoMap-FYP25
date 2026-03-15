"""
Validation Module
=================
Quality control checks for embeddings and pipeline outputs.

Functions:
    - validate_complete(): Run all QC checks
    - ValidationReport: Structured report of all QC results
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured validation report."""
    is_valid: bool
    checks: Dict[str, bool]
    issues: Dict[str, str]
    metrics: Dict[str, float]


def validate_complete(
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    barcodes: np.ndarray,
    check_nan: bool = True,
    check_inf: bool = True,
    nan_threshold: float = 0.01
) -> ValidationReport:
    """
    Run comprehensive validation checks.
    
    Args:
        embeddings_dict: Dict of modality -> embeddings
        labels: Ground truth labels
        barcodes: Spot barcodes
        check_nan: Check for NaN values
        check_inf: Check for infinite values
        nan_threshold: Max allowed NaN fraction
        
    Returns:
        ValidationReport with results
    """
    checks = {}
    issues = {}
    metrics = {}
    
    # Check 1: Shape consistency
    try:
        n_samples = labels.shape[0]
        assert len(barcodes) == n_samples, f"Barcode count mismatch: {len(barcodes)} vs {n_samples}"
        for modality, emb in embeddings_dict.items():
            assert emb.shape[0] == n_samples, f"{modality} shape mismatch: {emb.shape[0]} vs {n_samples}"
        checks["shape_consistency"] = True
        metrics["n_samples"] = n_samples
    except AssertionError as e:
        checks["shape_consistency"] = False
        issues["shape_consistency"] = str(e)
    
    # Check 2: NaN values
    if check_nan:
        try:
            nan_found = False
            for modality, emb in embeddings_dict.items():
                nan_frac = np.isnan(emb).sum() / emb.size
                metrics[f"{modality}_nan_fraction"] = nan_frac
                if nan_frac > nan_threshold:
                    nan_found = True
                    issues[f"{modality}_nan"] = f"NaN fraction: {nan_frac:.4f}"
            checks["nan_check"] = not nan_found
        except Exception as e:
            checks["nan_check"] = False
            issues["nan_check"] = str(e)
    
    # Check 3: Infinite values
    if check_inf:
        try:
            inf_found = False
            for modality, emb in embeddings_dict.items():
                inf_count = np.isinf(emb).sum()
                metrics[f"{modality}_inf_count"] = inf_count
                if inf_count > 0:
                    inf_found = True
                    issues[f"{modality}_inf"] = f"Infinite values found: {inf_count}"
            checks["inf_check"] = not inf_found
        except Exception as e:
            checks["inf_check"] = False
            issues["inf_check"] = str(e)
    
    # Determine overall validity
    is_valid = all(checks.values()) if checks else True
    
    if not is_valid:
        logger.warning("Validation failed with issues:")
        for issue, msg in issues.items():
            logger.warning(f"  - {issue}: {msg}")
    else:
        logger.info("✓ All validation checks passed")
    
    return ValidationReport(
        is_valid=is_valid,
        checks=checks,
        issues=issues,
        metrics=metrics
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example
    embeddings = {
        "UNI": np.random.randn(100, 150).astype(np.float32),
        "scVI": np.random.randn(100, 128).astype(np.float32)
    }
    labels = np.random.randint(0, 5, 100)
    barcodes = np.array([f"SPOT_{i}" for i in range(100)])
    
    report = validate_complete(embeddings, labels, barcodes)
    print(f"Valid: {report.is_valid}")
    print(f"Metrics: {report.metrics}")
