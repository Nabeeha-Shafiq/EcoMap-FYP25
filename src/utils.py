"""
Utilities Module
================
Common helper functions used across the pipeline.

Includes:
    - Logging setup
    - Color constants for visualizations
    - Metric calculations
    - File I/O helpers
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# ============ LOGGING ============

def setup_logging(
    log_dir: str = "./logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("modular_pipeline")
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============ COLORS ============

# Vivid, high-contrast colors for ecotypes
ECOTYPE_COLORS = {
    "Fibrotic": "#FF0000",              # Vivid Red
    "Immunosuppressive": "#0047AB",     # Vivid Blue
    "Invasive_Border": "#00AA00",       # Vivid Green
    "Metabolic": "#AA00AA",             # Vivid Purple
    "Normal_Adjacent": "#FF6600"        # Vivid Orange
}

# For 0-indexed class assignments
ECOTYPE_COLORS_INDEXED = [
    "#FF0000",      # 0: Vivid Red
    "#0047AB",      # 1: Vivid Blue
    "#00AA00",      # 2: Vivid Green
    "#AA00AA",      # 3: Vivid Purple
    "#FF6600"       # 4: Vivid Orange
]

# RGB versions (0-1 scale for matplotlib)
ECOTYPE_COLORS_RGB = [
    (1.0, 0.0, 0.0),       # Red
    (0.0, 0.29, 0.67),     # Blue
    (0.0, 0.67, 0.0),      # Green
    (0.67, 0.0, 0.67),     # Purple
    (1.0, 0.4, 0.0)        # Orange
]


def get_ecotype_color(ecotype_name: str) -> str:
    """
    Get hex color for ecotype name.
    
    Args:
        ecotype_name: Ecotype name (e.g., 'Fibrotic')
        
    Returns:
        Hex color code
    """
    return ECOTYPE_COLORS.get(ecotype_name, "#808080")  # Gray fallback


def get_color_by_index(index: int) -> str:
    """
    Get hex color by class index.
    
    Args:
        index: Class index (0-4)
        
    Returns:
        Hex color code
    """
    if 0 <= index < len(ECOTYPE_COLORS_INDEXED):
        return ECOTYPE_COLORS_INDEXED[index]
    return "#808080"  # Gray fallback


# ============ METRICS ============

def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Calculate per-class precision, recall, F1.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dict of metrics per class
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    metrics = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i])
        }
    
    return metrics


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix [N_classes, N_classes]
    """
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)


# ============ FILE HANDLING ============

def ensure_directory(dir_path: str) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return Path(file_path).stat().st_size / (1024**2)


# ============ DATA HANDLING ============

def normalize_embeddings(
    embeddings: np.ndarray,
    method: str = "z-score"
) -> np.ndarray:
    """
    Normalize embeddings.
    
    Args:
        embeddings: Embeddings array [N_samples, N_dims]
        method: 'z-score' or 'min-max'
        
    Returns:
        Normalized embeddings
    """
    if method == "z-score":
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        return (embeddings - mean) / (std + 1e-8)
    elif method == "min-max":
        min_val = embeddings.min(axis=0)
        max_val = embeddings.max(axis=0)
        return (embeddings - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        labels: Class labels [N_samples]
        n_classes: Number of classes
        
    Returns:
        Weight for each class
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    weights = compute_class_weight(
        'balanced',
        classes=np.arange(n_classes),
        y=labels
    )
    return weights.astype(np.float32)


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Utilities loaded")
    print(f"Ecotype colors: {ECOTYPE_COLORS}")
