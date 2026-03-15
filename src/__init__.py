"""
EcoMap Modular Pipeline
=======================
Unified training pipeline for spatial transcriptomics ecotype classification.

Enables training on any spatial transcriptomics dataset (GEO, Zenodo, or custom)
by only changing the YAML configuration file.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FYP Team"

from .config_loader import load_config, PipelineConfig
from .embeddings import (
    auto_detect_embedding_files,
    load_embedding_files,
    load_labels_and_metadata,
    validate_barcode_alignment,
    create_hdf5_dataset,
    load_hdf5_dataset
)
from .validation import validate_complete, ValidationReport

__all__ = [
    "load_config",
    "PipelineConfig",
    "auto_detect_embedding_files",
    "load_embedding_files",
    "load_labels_and_metadata",
    "validate_barcode_alignment",
    "create_hdf5_dataset",
    "load_hdf5_dataset",
    "validate_complete",
    "ValidationReport",
]
