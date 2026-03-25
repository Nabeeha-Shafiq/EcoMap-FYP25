#!/usr/bin/env python3
"""
Input Data Loader: Convert CSV Embeddings to NumPy Arrays
===========================================================

This script loads the three modality CSV files from the input_dataset folder
and converts them to NumPy arrays for processing within the pipeline.

Input Format (from input_dataset folder):
    - cell-composition-embeddings.csv: barcode | embedding_1 | ... | embedding_25
    - gene_embedding_combined.csv: barcode | embedding_1 | ... | embedding_128
    - image_encoder_embeddings.csv: barcode | embedding_1 | ... | embedding_150

Output (as NumPy arrays):
    - uni_embeddings.npy: (N_samples, 150) - Image encoder embeddings
    - scvi_embeddings.npy: (N_samples, 128) - Gene encoder embeddings
    - rctd_embeddings.npy: (N_samples, 25) - Cell composition embeddings
    - barcodes.npy: (N_samples,) - Barcodes with patient prefix
    - Validation report showing quality checks

Features:
    ✓ Load 3 separate CSV files (cell, gene, image)
    ✓ Auto-detect embedding dimensions
    ✓ Validate barcode alignment across modalities
    ✓ Handle patient-prefixed barcodes (P1_, P2_, etc.)
    ✓ Check for NaN/Inf values
    ✓ Generate comprehensive validation report
    ✓ Save as NumPy arrays for efficient pipeline processing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from datetime import datetime
import yaml

print("\n" + "="*100)
print("INPUT DATA LOADER: CSV to NumPy Conversion")
print("="*100 + "\n")

def load_embeddings_csv(csv_path, modality_name):
    """
    Load CSV embedding file and extract embeddings.
    
    Format: barcode | embedding_1 | embedding_2 | ... | embedding_N
    """
    print(f"  Loading {modality_name}...", end=" ")
    
    df = pd.read_csv(csv_path)
    
    # First column should be barcodes
    barcodes = df.iloc[:, 0].values
    
    # Rest are embeddings
    embeddings = df.iloc[:, 1:].values.astype(np.float32)
    
    n_samples, n_dims = embeddings.shape
    print(f"✓ ({n_samples}, {n_dims})")
    
    return barcodes, embeddings, n_dims

def validate_barcode_alignment(uni_barcodes, scvi_barcodes, rctd_barcodes, uni_emb, scvi_emb, rctd_emb):
    """
    Validate that all modalities have the same barcodes in the same order.
    Filter to common barcodes if mismatches detected.
    Returns (uni_barcodes, scvi_barcodes, rctd_barcodes, uni_emb, scvi_emb, rctd_emb)
    ALL aligned to common barcodes.
    """
    print("\n[STAGE 2] Validating Barcode Alignment")
    print("─" * 80)
    
    all_match = np.array_equal(uni_barcodes, scvi_barcodes) and np.array_equal(scvi_barcodes, rctd_barcodes)
    
    print(f"  UNI samples: {len(uni_barcodes):,}")
    print(f"  scVI samples: {len(scvi_barcodes):,}")
    print(f"  RCTD samples: {len(rctd_barcodes):,}")
    print(f"  All barcodes identical: {'YES ✓' if all_match else 'NO ✗'}")
    
    if not all_match:
        print("  ⚠ WARNING: Barcodes do not match! Filtering to common barcodes...")
        uni_set = set(uni_barcodes)
        scvi_set = set(scvi_barcodes)
        rctd_set = set(rctd_barcodes)
        
        common = sorted(uni_set & scvi_set & rctd_set)
        print(f"    Common barcodes: {len(common):,}")
        
        if len(common) < len(uni_set) * 0.95:
            print(f"    ✗ ERROR: Less than 95% overlap detected!")
            return None, None, None, None, None, None
        
        # Filter embeddings to common barcodes
        common_set = set(common)
        
        # Find indices in each original array
        uni_idx = [i for i, bc in enumerate(uni_barcodes) if bc in common_set]
        scvi_idx = [i for i, bc in enumerate(scvi_barcodes) if bc in common_set]
        rctd_idx = [i for i, bc in enumerate(rctd_barcodes) if bc in common_set]
        
        # Filter embeddings and barcodes
        uni_barcodes = uni_barcodes[uni_idx]
        scvi_barcodes = scvi_barcodes[scvi_idx]
        rctd_barcodes = rctd_barcodes[rctd_idx]
        
        uni_emb = uni_emb[uni_idx]
        scvi_emb = scvi_emb[scvi_idx]
        rctd_emb = rctd_emb[rctd_idx]
        
        # Verify alignment after filtering
        if not (np.array_equal(uni_barcodes, scvi_barcodes) and np.array_equal(scvi_barcodes, rctd_barcodes)):
            print("    ✗ ERROR: Barcodes still misaligned after filtering!")
            return None, None, None, None, None, None
        
        print(f"    ✓ Aligned to {len(uni_barcodes):,} common barcodes")
    
    print()
    return uni_barcodes, scvi_barcodes, rctd_barcodes, uni_emb, scvi_emb, rctd_emb

def check_data_quality(uni_emb, scvi_emb, rctd_emb, barcodes):
    """
    Check for NaN, Inf, and zero values.
    """
    print("[STAGE 3] Data Quality Checks")
    print("─" * 80)
    
    checks = {
        'UNI': {
            'nan': np.sum(np.isnan(uni_emb)),
            'inf': np.sum(np.isinf(uni_emb)),
            'zero': np.sum(uni_emb == 0),
            'total_elements': uni_emb.size
        },
        'scVI': {
            'nan': np.sum(np.isnan(scvi_emb)),
            'inf': np.sum(np.isinf(scvi_emb)),
            'zero': np.sum(scvi_emb == 0),
            'total_elements': scvi_emb.size
        },
        'RCTD': {
            'nan': np.sum(np.isnan(rctd_emb)),
            'inf': np.sum(np.isinf(rctd_emb)),
            'zero': np.sum(rctd_emb == 0),
            'total_elements': rctd_emb.size
        },
        'Barcodes': {
            'nan': np.sum(pd.isna(barcodes)),
            'duplicate': len(barcodes) - len(np.unique(barcodes))
        }
    }
    
    all_pass = True
    for modality, metrics in checks.items():
        print(f"\n  {modality}:")
        if modality != 'Barcodes':
            nan_pct = (metrics['nan'] / metrics['total_elements']) * 100
            inf_pct = (metrics['inf'] / metrics['total_elements']) * 100
            zero_pct = (metrics['zero'] / metrics['total_elements']) * 100
            
            nan_pass = metrics['nan'] == 0
            inf_pass = metrics['inf'] == 0
            
            print(f"    NaN values: {metrics['nan']:,} ({nan_pct:.2f}%) {'✓' if nan_pass else '✗'}")
            print(f"    Inf values: {metrics['inf']:,} ({inf_pct:.2f}%) {'✓' if inf_pass else '✗'}")
            print(f"    Zero values: {metrics['zero']:,} ({zero_pct:.2f}%)")
            
            if not (nan_pass and inf_pass):
                all_pass = False
        else:
            dup_pass = metrics['duplicate'] == 0
            print(f"    Total barcodes: {len(barcodes):,}")
            print(f"    Unique barcodes: {len(np.unique(barcodes)):,}")
            print(f"    Duplicate barcodes: {metrics['duplicate']} {'✓' if dup_pass else '✗'}")
            print(f"    NaN barcodes: {metrics['nan']} {'✓' if metrics['nan'] == 0 else '✗'}")
            
            if not dup_pass or metrics['nan'] > 0:
                all_pass = False
    
    print()
    return all_pass

def main():
    parser = argparse.ArgumentParser(
        description="Load input embeddings from CSV and convert to NumPy arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to config.yaml file (required - contains all input file paths)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for NumPy files"
    )
    
    # Legacy arguments (kept for backward compatibility, overridden by config)
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="[DEPRECATED] Path to input_dataset folder - use --config instead"
    )
    
    parser.add_argument(
        "--cell-file",
        type=str,
        default=None,
        help="[DEPRECATED] Cell composition CSV filename - use --config instead"
    )
    
    parser.add_argument(
        "--gene-file",
        type=str,
        default=None,
        help="[DEPRECATED] Gene expression CSV filename - use --config instead"
    )
    
    parser.add_argument(
        "--image-file",
        type=str,
        default=None,
        help="[DEPRECATED] Image encoder CSV filename - use --config instead"
    )
    
    args = parser.parse_args()
    
    # If config is provided, read all paths from it
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_dataset_config = config.get('input_dataset', {})
        
        # Extract paths from config - resolve relative to CWD (where pipeline is run)
        # Config paths are relative to the directory where run_pipeline.sh is executed
        image_file = Path(input_dataset_config.get('image_encoder_embeddings', ''))
        if not image_file.is_absolute():
            image_file = Path.cwd() / image_file
            
        gene_file = Path(input_dataset_config.get('gene_embeddings', ''))
        if not gene_file.is_absolute():
            gene_file = Path.cwd() / gene_file
            
        cell_file = Path(input_dataset_config.get('cell_composition_embeddings', ''))
        if not cell_file.is_absolute():
            cell_file = Path.cwd() / cell_file
        
        # Get directory from first file
        input_dir = image_file.parent
    else:
        # Fallback to command-line arguments (for backward compatibility)
        if not args.input_dir:
            print("  ✗ ERROR: Either --config or --input-dir must be provided")
            sys.exit(1)
        
        input_dir = Path(args.input_dir)
        
        image_file = input_dir / (args.image_file or "image_encoder_embeddings.csv")
        gene_file = input_dir / (args.gene_file or "gene_embedding_combined.csv")
        cell_file = input_dir / (args.cell_file or "cell-composition-embeddings.csv")
    
    output_dir = Path(args.output_dir)
    
    # Verify input files exist
    if not image_file.exists():
        print(f"  ✗ ERROR: File not found: {image_file}")
        sys.exit(1)
    if not gene_file.exists():
        print(f"  ✗ ERROR: File not found: {gene_file}")
        sys.exit(1)
    if not cell_file.exists():
        print(f"  ✗ ERROR: File not found: {cell_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if CSV files exist
    print("[STAGE 1] Loading CSV Files")
    print("─" * 80)
    
    # Load embeddings
    print(f"\n  From directory: {input_dir}")
    print()
    
    uni_barcodes, uni_emb, uni_dims = load_embeddings_csv(image_file, "UNI (Image)")
    scvi_barcodes, scvi_emb, scvi_dims = load_embeddings_csv(gene_file, "scVI (Gene)")
    rctd_barcodes, rctd_emb, rctd_dims = load_embeddings_csv(cell_file, "RCTD (Cell)")
    
    # Validate alignment AND filter to common barcodes
    result = validate_barcode_alignment(uni_barcodes, scvi_barcodes, rctd_barcodes, uni_emb, scvi_emb, rctd_emb)
    if result[0] is None:
        print("  ✗ ERROR: Barcode validation/filtering failed!")
        sys.exit(1)
    
    uni_barcodes, scvi_barcodes, rctd_barcodes, uni_emb, scvi_emb, rctd_emb = result
    
    # Data quality checks
    quality_pass = check_data_quality(uni_emb, scvi_emb, rctd_emb, uni_barcodes)
    
    print("[STAGE 4] Saving NumPy Arrays")
    print("─" * 80)
    
    # Save NumPy arrays
    try:
        np.save(output_dir / "uni_embeddings.npy", uni_emb)
        print(f"  ✓ Saved: uni_embeddings.npy {uni_emb.shape}")
        
        np.save(output_dir / "scvi_embeddings.npy", scvi_emb)
        print(f"  ✓ Saved: scvi_embeddings.npy {scvi_emb.shape}")
        
        np.save(output_dir / "rctd_embeddings.npy", rctd_emb)
        print(f"  ✓ Saved: rctd_embeddings.npy {rctd_emb.shape}")
        
        np.save(output_dir / "barcodes.npy", uni_barcodes)
        print(f"  ✓ Saved: barcodes.npy {uni_barcodes.shape}")
        
        print()
    except Exception as e:
        print(f"  ✗ ERROR saving files: {e}")
        sys.exit(1)
    
    # Summary
    print("[SUMMARY]")
    print("─" * 80)
    print(f"  Total samples: {len(uni_barcodes):,}")
    print(f"  UNI dimensions: {uni_dims}")
    print(f"  scVI dimensions: {scvi_dims}")
    print(f"  RCTD dimensions: {rctd_dims}")
    print(f"  Total fused dimensions: {uni_dims + scvi_dims + rctd_dims}")
    print(f"  Barcode format: Patient-prefixed (e.g., P1_AAACAACGAATAGTTC-1)")
    print(f"  Data Quality: {'PASS ✓' if quality_pass else 'REVIEW NEEDED ⚠'}")
    print(f"  Output directory: {output_dir}")
    print()
    print("="*100)
    print("✓ INPUT LOADING COMPLETE")
    print("="*100 + "\n")
    
    return 0 if quality_pass else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
