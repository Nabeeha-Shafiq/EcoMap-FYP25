"""
Embeddings Module
=================
Handles loading multimodal embeddings from various file formats (NPY, CSV),
validating barcode alignment, and converting to HDF5 for efficient storage.

Key Functions:
    - auto_detect_embedding_files(): Find embedding files by modality name
    - load_embedding_files(): Load NPY/CSV files and align by barcode
    - load_labels_and_metadata(): Load ground truth labels
    - validate_barcode_alignment(): Ensure all modalities have same spots
    - create_hdf5_dataset(): Convert to HDF5 single file
    - load_hdf5_dataset(): Read HDF5 back into memory

HDF5 Structure:
    /embeddings/
        /UNI          [N_samples, 150]
        /scVI         [N_samples, 128]
        /RCTD         [N_samples, 25]
    /labels           [N_samples]
    /barcodes         [N_samples] (str)
    /spatial          [N_samples, 2] (optional)
    /metadata         (attrs)
        - dataset_name
        - n_samples
        - n_features_total
        - modalities
        - creation_date
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)


def auto_detect_embedding_files(
    base_dir: str,
    modality_name: str,
    allowed_formats: List[str] = ["npy", "csv", "xlsx"]
) -> Optional[str]:
    """
    Auto-detect embedding file for given modality.
    
    Searches for files containing the modality name (case-insensitive) with
    allowed formats. Returns first match.
    
    Args:
        base_dir: Directory to search in
        modality_name: Modality name (e.g., 'UNI', 'scVI', 'RCTD')
        allowed_formats: File formats to accept
        
    Returns:
        Path to embedding file, or None if not found
        
    Example:
        >>> auto_detect_embedding_files("/data", "UNI")
        "/data/UNI_embeddings.npy"
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return None
    
    modality_lower = modality_name.lower()
    
    for format_ext in allowed_formats:
        # Search for files with modality name and format
        for file_path in base_dir.rglob(f"*.{format_ext}"):
            if modality_lower in file_path.name.lower():
                logger.info(f"Auto-detected {modality_name}: {file_path}")
                return str(file_path)
    
    logger.warning(f"Could not auto-detect embedding file for {modality_name} in {base_dir}")
    return None


def load_embedding_file(
    file_path: str,
    file_format: str = "csv",
    expected_dim: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load embedding file in CSV format (barcode column + embeddings).
    
    If CSV, returns both embeddings and barcodes extracted from first column.
    
    Args:
        file_path: Path to embedding CSV file (barcode + embeddings)
        file_format: 'csv' (barcode as first column, then embeddings)
        expected_dim: If provided, validate embedding dimension matches (excluding barcode)
        
    Returns:
        Tuple of (embeddings [N_samples, N_dims], barcodes [N_samples])
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format is invalid or dimension mismatch
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
            
            # First column is barcode
            barcodes = df.iloc[:, 0].values
            
            # Remaining columns are embeddings
            embeddings = df.iloc[:, 1:].values.astype(np.float32)
            
        else:
            raise ValueError(f"Unsupported format: {file_format}. Expected 'csv' with barcode column.")
        
        # Validate dimension if provided
        if expected_dim is not None:
            actual_dim = embeddings.shape[1]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch for {file_path}: expected {expected_dim}, got {actual_dim}"
                )
        
        logger.info(f"Loaded embeddings from {file_path}: shape {embeddings.shape}, barcodes {len(barcodes)}")
        return embeddings, barcodes
    
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def load_labels_file(
    file_path: str,
    file_format: str = "csv"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth labels from CSV (barcode + label columns).
    
    Returns both labels and barcodes for alignment verification.
    
    Args:
        file_path: Path to labels CSV file (barcode + ecotype/label)
        file_format: 'csv' (barcode as first column, label in second)
        
    Returns:
        Tuple of (labels [N_samples], barcodes [N_samples])
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Labels file not found: {file_path}")
    
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
            
            # First column is barcode
            barcodes = df.iloc[:, 0].values
            
            # Second column is label/ecotype
            labels = df.iloc[:, 1].values
            
            # Convert to integers if they are ecotypes (categorical)
            if isinstance(labels[0], str):
                # Map ecotype names to integers
                unique_labels = np.unique(labels)
                label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
                labels = np.array([label_to_int[label] for label in labels])
            else:
                labels = labels.astype(int)
            
        else:
            raise ValueError(f"Unsupported format: {file_format}. Expected 'csv'.")
        
        logger.info(f"Loaded labels from {file_path}: shape {labels.shape}, unique classes {len(np.unique(labels))}")
        return labels, barcodes
    
    except Exception as e:
        raise ValueError(f"Error loading labels {file_path}: {e}")


def load_barcodes(
    file_path: Optional[str],
    file_format: str = "csv"
) -> Optional[np.ndarray]:
    """
    Load spot barcodes (optional - usually extracted from embeddings/labels CSV).
    
    Args:
        file_path: Path to barcodes file (optional)
        file_format: 'csv'
        
    Returns:
        Barcodes array [N_samples] of strings, or None if file_path is None
    """
    if file_path is None:
        return None
        
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Barcodes file not found: {file_path}")
    
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
            # First column is barcode
            barcodes = df.iloc[:, 0].values
            barcodes = np.array([str(b) for b in barcodes])
        elif file_format.lower() == "npy":
            barcodes = np.load(file_path)
            if barcodes.dtype.kind == 'U':  # Unicode string
                barcodes = np.array([str(b) for b in barcodes])
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded {len(barcodes)} barcodes from {file_path}")
        return barcodes
    
    except Exception as e:
        raise ValueError(f"Error loading barcodes {file_path}: {e}")



def load_spatial_coordinates(
    file_path: str,
    file_format: str = "csv"
) -> Optional[np.ndarray]:
    """
    Load spatial coordinates (if provided).
    
    Args:
        file_path: Path to spatial file
        file_format: 'csv' or 'npy'
        
    Returns:
        Spatial array [N_samples, 2] or None
    """
    if file_path is None:
        return None
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Spatial file not found: {file_path}")
        return None
    
    try:
        if file_format.lower() == "csv":
            spatial = pd.read_csv(file_path, index_col=0).values
        elif file_format.lower() == "npy":
            spatial = np.load(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded spatial coordinates: shape {spatial.shape}")
        return spatial
    
    except Exception as e:
        logger.warning(f"Error loading spatial coordinates: {e}")
        return None


def load_patient_mapping(
    file_path: Optional[str],
    file_format: str = "csv"
) -> Optional[np.ndarray]:
    """
    Load patient-to-barcode mapping (if provided).
    
    Used for visualization modules to plot per-patient analyses.
    
    Args:
        file_path: Path to patient mapping file (barcode | patient_id)
        file_format: 'csv'
        
    Returns:
        Patient IDs array [N_samples] (strings), or None if not provided
    """
    if file_path is None:
        return None
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Patient mapping file not found: {file_path}")
        return None
    
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
            # First column is barcode, second column is patient_id
            patient_ids = df.iloc[:, 1].values
            patient_ids = np.array([str(p) for p in patient_ids])
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded patient mapping: {len(patient_ids)} samples, {len(np.unique(patient_ids))} unique patients")
        return patient_ids
    
    except Exception as e:
        logger.warning(f"Error loading patient mapping: {e}")
        return None


def validate_barcode_alignment(
    barcodes_dict: Dict[str, np.ndarray],
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray
) -> Tuple[bool, str]:
    """
    Validate that all modalities have the same barcodes in the same order.
    
    Args:
        barcodes_dict: Dict of modality -> barcodes array
        embeddings_dict: Dict of modality -> embeddings array
        labels: Ground truth labels
        
    Returns:
        (is_aligned, message): Boolean indicating alignment and status message
    """
    if len(barcodes_dict) == 0:
        return False, "No barcodes provided"
    
    # Get first barcode array as reference
    ref_modality = list(barcodes_dict.keys())[0]
    ref_barcodes = barcodes_dict[ref_modality]
    
    # Check all modalities have same barcode count
    for modality, barcodes in barcodes_dict.items():
        if len(barcodes) != len(ref_barcodes):
            msg = f"Barcode count mismatch for {modality}: {len(barcodes)} vs {len(ref_barcodes)}"
            logger.error(msg)
            return False, msg
    
    # Check all embeddings have same sample count
    for modality, embeddings in embeddings_dict.items():
        if embeddings.shape[0] != len(ref_barcodes):
            msg = f"Embedding sample count mismatch for {modality}: {embeddings.shape[0]} vs {len(ref_barcodes)}"
            logger.error(msg)
            return False, msg
    
    # Check labels have same sample count
    if len(labels) != len(ref_barcodes):
        msg = f"Labels sample count mismatch: {len(labels)} vs {len(ref_barcodes)}"
        logger.error(msg)
        return False, msg
    
    # Check all barcodes match (if provided)
    for modality, barcodes in barcodes_dict.items():
        if modality != ref_modality:
            if not np.array_equal(barcodes, ref_barcodes):
                msg = f"Barcode mismatch between {ref_modality} and {modality}"
                logger.error(msg)
                return False, msg
    
    msg = f"✓ Barcode alignment valid: {len(ref_barcodes)} spots across {len(embeddings_dict)} modalities"
    logger.info(msg)
    return True, msg


def check_data_quality(
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    nan_threshold: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Check for NaN, Inf, and other data quality issues.
    
    Args:
        embeddings_dict: Dict of modality -> embeddings
        labels: Ground truth labels
        nan_threshold: Maximum allowed fraction of NaN values
        
    Returns:
        (is_valid, report): Boolean and detailed report
    """
    report = {}
    all_valid = True
    
    for modality, embeddings in embeddings_dict.items():
        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()
        nan_frac = nan_count / embeddings.size
        
        report[modality] = {
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "nan_fraction": float(nan_frac),
            "valid": nan_frac <= nan_threshold and inf_count == 0
        }
        
        if not report[modality]["valid"]:
            all_valid = False
            logger.warning(
                f"{modality}: NaN={nan_count} ({nan_frac:.4f}), Inf={inf_count}"
            )
        else:
            logger.info(f"✓ {modality}: No NaN/Inf issues")
    
    # Check labels
    nan_count_labels = np.isnan(labels).sum() if labels.dtype.kind == 'f' else 0
    report["labels"] = {"nan_count": int(nan_count_labels), "valid": nan_count_labels == 0}
    
    return all_valid, report


def create_hdf5_dataset(
    output_path: str,
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    barcodes: np.ndarray,
    spatial: Optional[np.ndarray] = None,
    patient_mapping: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create HDF5 file from embeddings, labels, and metadata.
    
    Args:
        output_path: Path to output HDF5 file
        embeddings_dict: Dict of modality name -> embeddings array
        labels: Ground truth labels [N_samples]
        barcodes: Spot barcodes [N_samples]
        spatial: Optional spatial coordinates [N_samples, 2]
        patient_mapping: Optional patient IDs [N_samples]
        metadata: Optional metadata dict
        
    Returns:
        Path to created HDF5 file
        
    Raises:
        ValueError: If data shapes are inconsistent
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate shapes
    n_samples = labels.shape[0]
    if len(barcodes) != n_samples:
        raise ValueError(f"Barcode count mismatch: {len(barcodes)} vs {n_samples}")
    
    for modality, embeddings in embeddings_dict.items():
        if embeddings.shape[0] != n_samples:
            raise ValueError(
                f"Embedding shape mismatch for {modality}: "
                f"{embeddings.shape[0]} vs {n_samples}"
            )
    
    try:
        with h5py.File(str(output_path), 'w') as f:
            # Create embeddings group
            emb_group = f.create_group('embeddings')
            for modality, embeddings in embeddings_dict.items():
                emb_group.create_dataset(
                    modality,
                    data=embeddings,
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4
                )
            
            # Create labels
            f.create_dataset(
                'labels',
                data=labels,
                dtype='int32',
                compression='gzip'
            )
            
            # Create barcodes (stored as fixed-length strings)
            # Convert to object array of strings to avoid h5py unicode issues
            barcode_list = [str(b) for b in barcodes]
            str_dtype = h5py.string_dtype(encoding='utf-8', length=None)
            f.create_dataset(
                'barcodes',
                data=np.array(barcode_list, dtype=object),
                dtype=str_dtype,
                compression='gzip'
            )
            
            # Create spatial if provided
            if spatial is not None:
                f.create_dataset(
                    'spatial',
                    data=spatial,
                    dtype='float32',
                    compression='gzip'
                )
            
            # Create patient mapping if provided
            if patient_mapping is not None:
                patient_list = [str(p) for p in patient_mapping]
                str_dtype = h5py.string_dtype(encoding='utf-8', length=None)
                f.create_dataset(
                    'patient_mapping',
                    data=np.array(patient_list, dtype=object),
                    dtype=str_dtype,
                    compression='gzip'
                )
            
            # Create metadata attributes
            f.attrs['n_samples'] = n_samples
            f.attrs['n_modalities'] = len(embeddings_dict)
            f.attrs['modalities'] = list(embeddings_dict.keys())
            f.attrs['creation_date'] = datetime.now().isoformat()
            
            # Add custom metadata
            if metadata is not None:
                for key, value in metadata.items():
                    try:
                        f.attrs[key] = value
                    except TypeError:
                        # If value is not HDF5 serializable, convert to string
                        f.attrs[key] = str(value)
        
        logger.info(f"✓ Created HDF5 file: {output_path}")
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        logger.info(f"  - Samples: {n_samples}")
        logger.info(f"  - Modalities: {list(embeddings_dict.keys())}")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error creating HDF5 file: {e}")
        raise


def load_hdf5_dataset(
    hdf5_path: str,
    modalities: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load data from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        modalities: If provided, only load specific modalities
        
    Returns:
        (embeddings_dict, labels, barcodes, spatial):
            - embeddings_dict: Dict of modality -> embeddings
            - labels: Ground truth labels
            - barcodes: Spot barcodes
            - spatial: Spatial coordinates (or None)
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    embeddings_dict = {}
    labels = None
    barcodes = None
    spatial = None
    
    try:
        with h5py.File(str(hdf5_path), 'r') as f:
            # Load embeddings
            if 'embeddings' in f:
                emb_group = f['embeddings']
                available_modalities = list(emb_group.keys())
                
                if modalities is None:
                    modalities = available_modalities
                
                for modality in modalities:
                    if modality in emb_group:
                        embeddings_dict[modality] = emb_group[modality][:]
                    else:
                        logger.warning(f"Modality {modality} not found in HDF5")
            
            # Load labels
            if 'labels' in f:
                labels = f['labels'][:]
            
            # Load barcodes
            if 'barcodes' in f:
                barcodes = f['barcodes'][:]
            
            # Load spatial (optional)
            if 'spatial' in f:
                spatial = f['spatial'][:]
            
            # Log metadata
            logger.info(f"Loaded HDF5 from {hdf5_path}")
            logger.info(f"  - Samples: {labels.shape[0]}")
            logger.info(f"  - Modalities: {list(embeddings_dict.keys())}")
            logger.info(f"  - Creation date: {f.attrs.get('creation_date', 'unknown')}")
    
    except Exception as e:
        raise ValueError(f"Error loading HDF5 file: {e}")
    
    return embeddings_dict, labels, barcodes, spatial


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create HDF5
    embeddings = {
        "UNI": np.random.randn(100, 150).astype(np.float32),
        "scVI": np.random.randn(100, 128).astype(np.float32),
        "RCTD": np.random.randn(100, 25).astype(np.float32)
    }
    labels = np.random.randint(0, 5, 100)
    barcodes = np.array([f"SPOT_{i:06d}" for i in range(100)])
    
    output_file = create_hdf5_dataset(
        "/tmp/test_embeddings.h5",
        embeddings,
        labels,
        barcodes,
        metadata={"dataset_name": "test", "version": "1.0"}
    )
    
    # Load back
    emb_loaded, labels_loaded, barcodes_loaded, _ = load_hdf5_dataset(output_file)
    print(f"✓ Round-trip successful: {list(emb_loaded.keys())}")
