#!/usr/bin/env python3
"""
Script 1: Convert Embeddings to HDF5
====================================
Entry point for converting multimodal embeddings from CSV format
into a single HDF5 file for efficient storage and loading.

This is the FIRST step in the modular pipeline.

Input Format (CSV):
    - image_encoder.csv: barcode | embedding_1 | embedding_2 | ... | embedding_N
    - cell_encoder.csv: barcode | embedding_1 | embedding_2 | ... | embedding_M
    - gene_encoder.csv: barcode | embedding_1 | embedding_2 | ... | embedding_K
    - labels.csv: barcode | ecotype
    - (optional) patient_mapping.csv: barcode | patient_id

Usage:
    python scripts/1_convert_embeddings.py --config config/geo_example.yaml
    python scripts/1_convert_embeddings.py --config config/zenodo_example.yaml

Features:
    ✓ Parse CSV with barcode column + embeddings
    ✓ Auto-detect embedding dimensions (if not specified)
    ✓ Validate barcode alignment across modalities
    ✓ Check for NaN/Inf values
    ✓ Create single HDF5 file with metadata
    ✓ Comprehensive error handling and logging

Output:
    - Single HDF5 file with structure:
        /embeddings/image_encoder    [N, 150]
        /embeddings/cell_encoder     [N, 128]
        /embeddings/gene_encoder     [N, 25]
        /labels            [N]
        /barcodes          [N]
        /spatial           [N, 2] (optional)
        /metadata          (attributes)
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import load_config, print_config
from embeddings import (
    load_embedding_file,
    load_labels_file,
    load_barcodes,
    load_spatial_coordinates,
    validate_barcode_alignment,
    check_data_quality,
    create_hdf5_dataset
)
from validation import validate_complete
from utils import setup_logging, ensure_directory


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert embeddings to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/1_convert_embeddings.py --config config/geo_example.yaml
    python scripts/1_convert_embeddings.py --config config/zenodo_example.yaml --output-dir ./data/geo_hdf5
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config (optional)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Specify exact HDF5 output filename (optional)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data quality checks (not recommended)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    
    try:
        logger.info("="*80)
        logger.info("MODULAR PIPELINE - STEP 1: Convert Embeddings to HDF5")
        logger.info("="*80)
        
        # ===== LOAD CONFIGURATION =====
        logger.info("\n[1/5] Loading configuration...")
        config = load_config(args.config)
        print_config(config)
        
        # ===== LOAD EMBEDDINGS =====
        logger.info("\n[2/5] Loading embedding modalities...")
        embeddings_dict = {}
        barcodes_dict = {}  # Store barcodes from each modality for alignment check
        
        for modality_name, emb_config in config.embeddings.items():
            try:
                logger.info(f"  Loading {modality_name}...")
                embeddings, barcodes = load_embedding_file(
                    file_path=emb_config.file_path,
                    file_format=emb_config.file_format,
                    expected_dim=emb_config.n_dims
                )
                embeddings_dict[modality_name] = embeddings
                barcodes_dict[modality_name] = barcodes
                logger.info(f"    ✓ Loaded: shape {embeddings.shape}, dtype {embeddings.dtype}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {modality_name}: {e}")
                raise
        
        # ===== LOAD LABELS AND METADATA =====
        logger.info("\n[3/5] Loading labels, barcodes, and metadata...")
        
        try:
            labels, labels_barcodes = load_labels_file(
                file_path=config.labels_metadata.labels_path,
                file_format=config.labels_metadata.labels_format
            )
            logger.info(f"  ✓ Labels: shape {labels.shape}, unique classes {len(np.unique(labels))}")
            
            # Use barcodes from labels file (primary source)
            barcodes = labels_barcodes
            barcodes_dict['labels'] = barcodes
            logger.info(f"  ✓ Barcodes (from labels): {len(barcodes)} samples")
            
            spatial = load_spatial_coordinates(
                file_path=config.labels_metadata.spatial_path,
                file_format="csv"
            )
            if spatial is not None:
                logger.info(f"  ✓ Spatial coordinates: shape {spatial.shape}")
            else:
                logger.info("  ⚠ Spatial coordinates not provided (optional)")
        
        except Exception as e:
            logger.error(f"Failed to load labels/metadata: {e}")
            raise
        
        # ===== VALIDATE DATA QUALITY =====
        if not args.skip_validation:
            logger.info("\n[4/5] Validating data quality...")
            
            # Barcode alignment
            is_aligned, msg = validate_barcode_alignment(
                barcodes_dict=barcodes_dict,
                embeddings_dict=embeddings_dict,
                labels=labels
            )
            logger.info(f"  {msg}")
            if not is_aligned:
                raise ValueError("Barcode alignment failed!")
            
            # NaN/Inf checks
            is_valid, report = check_data_quality(
                embeddings_dict=embeddings_dict,
                labels=labels,
                nan_threshold=config.validation.nan_threshold
            )
            logger.info(f"  Data quality check: {'✓ PASS' if is_valid else '✗ FAIL'}")
            
            for modality, quality in report.items():
                logger.debug(f"    {modality}: {quality}")
            
            if not is_valid and config.validation.check_nan:
                raise ValueError("Data quality checks failed!")
        
        else:
            logger.warning("  ⚠ Skipping data quality validation")
        
        # ===== CREATE HDF5 =====
        logger.info("\n[5/5] Creating HDF5 dataset...")
        
        # Determine output path
        output_dir = ensure_directory(args.output_dir or config.output_dir)
        
        if args.output_file:
            hdf5_path = output_dir / args.output_file
        else:
            hdf5_path = output_dir / f"{config.dataset.name}.h5"
        
        # Create metadata
        metadata = {
            "dataset_name": config.dataset.name,
            "description": config.dataset.description,
            "n_samples": labels.shape[0],
            "n_classes": config.dataset.n_classes,
            "class_names": ",".join(config.dataset.class_names),
            "modalities": ",".join(embeddings_dict.keys()),
            "features_per_modality": ",".join(
                [f"{m}:{config.embeddings[m].n_dims}" for m in embeddings_dict.keys()]
            ),
            "creation_date": datetime.now().isoformat(),
            "config_file": str(args.config)
        }
        
        # Add custom metadata
        if config.labels_metadata.metadata_fields:
            metadata.update(config.labels_metadata.metadata_fields)
        
        # Create HDF5
        hdf5_file = create_hdf5_dataset(
            output_path=str(hdf5_path),
            embeddings_dict=embeddings_dict,
            labels=labels,
            barcodes=barcodes,
            spatial=spatial,
            metadata=metadata
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ CONVERSION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Output file: {hdf5_file}")
        logger.info(f"File size: {Path(hdf5_file).stat().st_size / (1024**2):.2f} MB")
        logger.info(f"\nNext step: python scripts/2_validate_embeddings.py --input {hdf5_file}")
        logger.info(f"{'='*80}\n")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
