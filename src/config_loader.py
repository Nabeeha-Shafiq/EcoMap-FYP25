"""
Configuration Loader Module
============================
YAML configuration loading with Pydantic validation.

Enables dataset-agnostic pipeline by reading all parameters from a single YAML file.
Validates at startup to catch configuration errors before training begins.

Classes:
    - EmbeddingConfig: Paths and formats for each modality
    - DatasetConfig: Dataset metadata (name, n_samples, n_classes)
    - TrainingConfig: Model hyperparameters (hidden dims, lr, batch size)
    - ValidationConfig: QC parameters (nan threshold, correlation checks)
    - PipelineConfig: Complete configuration (top-level)

Usage:
    config = load_config("config/geo_example.yaml")
    print(f"Dataset: {config.dataset.name}")
    print(f"Embeddings: {config.embeddings.modalities}")
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding modality (image_encoder, cell_encoder, gene_encoder)."""
    
    name: str = Field(..., description="Modality name (e.g., 'image_encoder', 'cell_encoder', 'gene_encoder')")
    file_path: str = Field(..., description="Path to embedding CSV file (barcode column + embeddings)")
    file_format: str = Field(default="csv", description="File format: 'csv' only (barcode + embeddings)")
    n_dims: Optional[int] = Field(None, description="Expected dimension of embeddings (auto-detected if None)")
    description: Optional[str] = Field(None, description="Optional description of modality")
    
    @field_validator('file_format')
    @classmethod
    def validate_format(cls, v):
        """Ensure file format is 'csv' (barcode + embeddings)."""
        if v.lower() != "csv":
            raise ValueError("file_format must be 'csv' with barcode column as first column")
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "name": "image_encoder",
                "file_path": "/data/image_encoder.csv",
                "file_format": "csv",
                "n_dims": 150,
                "description": "Image encoder (morphology) embeddings: barcode + 150D"
            }
        }


class LabelsAndMetadataConfig(BaseModel):
    """Configuration for ground truth labels and metadata."""
    
    labels_path: str = Field(..., description="Path to ground truth labels CSV (barcode + ecotype)")
    labels_format: str = Field(default="csv", description="Format: 'csv' (barcode + label column)")
    barcode_path: Optional[str] = Field(None, description="Path to spot barcodes (optional if in labels)")
    barcode_format: str = Field(default="csv", description="Format: 'csv'")
    spatial_path: Optional[str] = Field(None, description="Path to spatial coordinates (.csv)")
    patient_mapping_path: Optional[str] = Field(None, description="Path to patient mapping (.csv)")
    metadata_fields: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "labels_path": "/data/labels.csv",
                "labels_format": "csv",
                "barcode_path": None,
                "spatial_path": "/data/spatial.csv",
                "patient_mapping_path": "/data/patient_mapping.csv",
                "metadata_fields": {"dataset_version": "v1.0", "tissue_type": "primary"}
            }
        }


class DatasetConfig(BaseModel):
    """Configuration for dataset properties."""
    
    name: str = Field(..., description="Dataset name (e.g., 'GEO', 'ZENODO')")
    n_samples: Optional[int] = Field(None, description="Expected number of samples (validation)")
    n_classes: int = Field(5, description="Number of ecotype classes")
    class_names: List[str] = Field(
        default=["Fibrotic", "Immunosuppressive", "Invasive_Border", "Metabolic", "Normal_Adjacent"],
        description="Class labels"
    )
    description: Optional[str] = Field(None, description="Dataset description")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "GEO_23342",
                "n_samples": 23342,
                "n_classes": 5,
                "class_names": ["Fibrotic", "Immunosuppressive", "Invasive_Border", "Metabolic", "Normal_Adjacent"],
                "description": "Primary tumor GEO dataset with 23.3K spots"
            }
        }


class TrainingConfig(BaseModel):
    """Configuration for model training hyperparameters."""
    
    hidden_dims: List[int] = Field(default=[256, 128, 64], description="MLP hidden layer sizes")
    n_epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=1e-5, description="L2 regularization")
    n_folds: int = Field(default=5, description="Number of cross-validation folds")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    use_class_weights: bool = Field(default=True, description="Balance class weights")
    early_stopping_patience: int = Field(default=15, description="Epochs without improvement before stopping")
    dropout_rate: float = Field(default=0.2, description="Dropout rate")
    
    class Config:
        schema_extra = {
            "example": {
                "hidden_dims": [256, 128, 64],
                "n_epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "n_folds": 5,
                "random_seed": 42,
                "use_class_weights": True,
                "early_stopping_patience": 15,
                "dropout_rate": 0.2
            }
        }


class ValidationConfig(BaseModel):
    """Configuration for data validation and quality checks."""
    
    check_nan: bool = Field(default=True, description="Check for NaN values")
    check_inf: bool = Field(default=True, description="Check for infinite values")
    check_correlation: bool = Field(default=True, description="Check cross-modality correlation")
    nan_threshold: float = Field(default=0.01, description="Allowed NaN fraction")
    barcode_alignment_required: bool = Field(default=True, description="Require exact barcode alignment")
    save_validation_report: bool = Field(default=True, description="Save QC report")
    
    class Config:
        schema_extra = {
            "example": {
                "check_nan": True,
                "check_inf": True,
                "check_correlation": True,
                "nan_threshold": 0.01,
                "barcode_alignment_required": True,
                "save_validation_report": True
            }
        }


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    embeddings: Dict[str, EmbeddingConfig] = Field(..., description="Embedding modalities")
    labels_metadata: LabelsAndMetadataConfig = Field(..., description="Labels and metadata")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training hyperparameters")
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation settings")
    output_dir: str = Field(default="./results", description="Output directory for results")
    input_hdf5_path: Optional[str] = Field(None, description="Path to pre-converted HDF5 file")
    
    @model_validator(mode='after')
    def validate_modalities(self):
        """Validate that embeddings dict keys match EmbeddingConfig names."""
        for key, config in self.embeddings.items():
            if key != config.name:
                raise ValueError(f"Embedding key '{key}' must match config.name '{config.name}'")
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "dataset": {
                    "name": "GEO_23342",
                    "n_samples": 23342,
                    "n_classes": 5
                },
                "embeddings": {
                    "image_encoder": {"name": "image_encoder", "file_path": "/data/image_encoder.csv", "file_format": "csv", "n_dims": 150},
                    "cell_encoder": {"name": "cell_encoder", "file_path": "/data/cell_encoder.csv", "file_format": "csv", "n_dims": 128},
                    "gene_encoder": {"name": "gene_encoder", "file_path": "/data/gene_encoder.csv", "file_format": "csv", "n_dims": 25}
                },
                "labels_metadata": {
                    "labels_path": "/data/labels.csv",
                    "spatial_path": "/data/spatial.csv"
                },
                "output_dir": "./results"
            }
        }


def load_config(config_path: str) -> PipelineConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        PipelineConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If configuration validation fails
        
    Example:
        >>> config = load_config("config/geo_example.yaml")
        >>> print(f"Dataset: {config.dataset.name}")
        >>> print(f"N samples: {config.dataset.n_samples}")
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    
    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    try:
        # Convert embedding configs
        if "embeddings" in config_dict:
            embeddings_raw = config_dict["embeddings"]
            embeddings = {}
            for name, config_dict_emb in embeddings_raw.items():
                config_dict_emb["name"] = name  # Ensure name field is set
                embeddings[name] = EmbeddingConfig(**config_dict_emb)
            config_dict["embeddings"] = embeddings
        
        # Convert labels/metadata config
        if "labels_metadata" in config_dict:
            config_dict["labels_metadata"] = LabelsAndMetadataConfig(**config_dict["labels_metadata"])
        
        # Convert dataset config
        if "dataset" in config_dict:
            config_dict["dataset"] = DatasetConfig(**config_dict["dataset"])
        
        # Convert training config (optional)
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        
        # Convert validation config (optional)
        if "validation" in config_dict:
            config_dict["validation"] = ValidationConfig(**config_dict["validation"])
        
        return PipelineConfig(**config_dict)
    
    except ValueError as e:
        raise ValueError(f"Configuration validation error: {e}")


def print_config(config: PipelineConfig) -> None:
    """
    Pretty-print configuration for debugging.
    
    Args:
        config: PipelineConfig object to print
    """
    print("\n" + "="*80)
    print("PIPELINE CONFIGURATION")
    print("="*80)
    print(f"\nDataset: {config.dataset.name}")
    print(f"  - Samples: {config.dataset.n_samples}")
    print(f"  - Classes: {config.dataset.class_names}")
    
    print(f"\nEmbedding Modalities:")
    for name, emb_config in config.embeddings.items():
        print(f"  - {name}: {emb_config.file_path} ({emb_config.n_dims}D, {emb_config.file_format.upper()})")
    
    print(f"\nLabels & Metadata:")
    print(f"  - Labels: {config.labels_metadata.labels_path}")
    print(f"  - Barcodes: {config.labels_metadata.barcode_path}")
    
    print(f"\nTraining:")
    print(f"  - Hidden dims: {config.training.hidden_dims}")
    print(f"  - Epochs: {config.training.n_epochs}")
    print(f"  - Folds: {config.training.n_folds}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    
    print(f"\nOutput: {config.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    config = load_config("config/geo_example.yaml")
    print_config(config)
