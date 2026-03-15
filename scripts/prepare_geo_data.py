#!/usr/bin/env python3
"""
Helper script to prepare GEO dataset for modular pipeline testing.

Creates concatenated embeddings, labels, barcodes, and spatial coordinates
from the existing GEO training data structure.

This is needed because the modular pipeline expects single concatenated files
per modality, while the existing GEO data is organized in per-patient folders.

Usage:
    python scripts/prepare_geo_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
FYP1_DIR = Path("/home/hp/Desktop/dataset_folder/FYP-1")
DATA_DIR = FYP1_DIR / "6-dec-training"
ISCHIA_FILE = FYP1_DIR / "ecotype_validation_final_ecotypes_25nov" / "ischia_with_validated_ecotypes.csv"
OUTPUT_DIR = FYP1_DIR / "prepared_for_modular_pipeline"
OUTPUT_DIR.mkdir(exist_ok=True)

def concatenate_patient_files(folder: Path, pattern: str, skip_index: bool = False) -> np.ndarray:
    """Find and concatenate patient files matching pattern."""
    files = sorted(list(folder.glob(f"{pattern}")))
    print(f"Found {len(files)} files matching {pattern}:")
    
    data_list = []
    for file in files:
        print(f"  - {file.name}")
        if file.suffix == ".npy":
            data = np.load(file)
        elif file.suffix == ".csv":
            df = pd.read_csv(file, index_col=0 if skip_index else None)
            data = df.values if skip_index else df.iloc[:, 1:].values
        else:
            continue
        data_list.append(data)
    
    combined = np.vstack(data_list)
    print(f"  → Combined shape: {combined.shape}\n")
    return combined


def prepare_geo_data():
    """Prepare GEO data for modular pipeline."""
    print("="*80)
    print("PREPARING GEO DATA FOR MODULAR PIPELINE")
    print("="*80 + "\n")
    
    # Load ISCHIA reference (contains ground truth labels & barcodes)
    print("Loading ISCHIA reference data (ground truth)...")
    ischia_df = pd.read_csv(ISCHIA_FILE, index_col=0)
    print(f"  - ISCHIA shape: {ischia_df.shape}")
    print(f"  - Columns: {list(ischia_df.columns)}")
    
    # Extract barcodes and labels
    barcodes = ischia_df.index.values
    print(f"  - Unique barcodes: {len(barcodes)}")
    
    # Map class names (need to check what labels are available)
    print(f"  - Unique validated ecotypes: {sorted(set(ischia_df['Validated_Ecotype']))}\n")
    
    # Create label mapping - for this test, we'll map ANY valid ecotype
    # Since only Fibrotic and Exocrine-like exist, we'll keep Fibrotic and skip Exocrine-like
    label_map = {
        'Fibrotic': 0,
        'Immunosuppressive': 1,
        'Invasive_Border': 2,
        'Metabolic': 3,
        'Normal_Adjacent': 4,
    }
    
    labels = np.array([label_map.get(x, -1) for x in ischia_df['Validated_Ecotype']])
    print(f"Mapping ecotypes to classes...")
    print(f"  - Using label_map: {label_map}")
    print(f"  - Initial labels shape: {labels.shape}")
    print(f"  - Labels distribution: {np.bincount(labels[labels >= 0])}\n")
    
    # Concatenate UNI embeddings
    print("Concatenating UNI embeddings...")
    uni_folder = DATA_DIR / "UNI_150D_Embeddings"
    uni_embeddings = concatenate_patient_files(uni_folder, "*UNI_150D.csv")
    print(f"  - UNI shape before filtering: {uni_embeddings.shape}")
    
    # Concatenate scVI embeddings
    print("Concatenating scVI embeddings...")
    scvi_folder = DATA_DIR / "scVI Embeddings"
    scvi_embeddings = concatenate_patient_files(scvi_folder, "*embedding.csv")
    print(f"  - scVI shape before filtering: {scvi_embeddings.shape}")
    
    # Concatenate RCTD embeddings
    print("Concatenating RCTD embeddings...")
    rctd_folder = DATA_DIR / "Cell Composition Embeddings"
    # RCTD file has barcodes in last column, so we need special handling
    rctd_files = sorted(list(rctd_folder.glob("*combined.csv")))
    rctd_data_list = []
    for file in rctd_files:
        print(f"  - {file.name}")
        df = pd.read_csv(file)
        # Remove barcode and patient columns (assumed to be last two)
        numeric_cols = [c for c in df.columns if c not in ['barcode', 'patient']]
        rctd_data_list.append(df[numeric_cols].values)
    rctd_embeddings = np.vstack(rctd_data_list)
    print(f"  → Combined shape: {rctd_embeddings.shape}\n")
    
    # Use embeddings shape as source of truth (it contains actual tissue spots)
    n_samples = uni_embeddings.shape[0]
    print(f"\nUsing embedding-based sample count: {n_samples}")
    
    # Trim labels and barcodes to match embedding size
    if len(labels) > n_samples:
        print(f"Trimming labels from {len(labels)} to {n_samples}")
        labels = labels[:n_samples]
        barcodes = barcodes[:n_samples]
    elif len(labels) < n_samples:
        print(f"WARNING: Fewer labels ({len(labels)}) than embeddings ({n_samples})")
        print("This might indicate a data alignment issue")
    
    # Trim scVI and RCTD to match UNI
    scvi_embeddings = scvi_embeddings[:n_samples]
    rctd_embeddings = rctd_embeddings[:n_samples]
    
    # Verify shapes match
    print("Verifying shapes match...")
    print(f"  - Labels: {labels.shape}")
    print(f"  - Barcodes: {barcodes.shape}")
    print(f"  - UNI: {uni_embeddings.shape}")
    print(f"  - scVI: {scvi_embeddings.shape}")
    print(f"  - RCTD: {rctd_embeddings.shape}\n")
    
    # Verify all have same number of samples
    assert labels.shape[0] == barcodes.shape[0] == uni_embeddings.shape[0] == scvi_embeddings.shape[0] == rctd_embeddings.shape[0], \
        f"Shape mismatch: labels={labels.shape[0]}, barcodes={barcodes.shape[0]}, uni={uni_embeddings.shape[0]}, scvi={scvi_embeddings.shape[0]}, rctd={rctd_embeddings.shape[0]}"
    
    # Trim dimensions to exact spec
    print("Trimming dimensions to spec...")
    min_uni = min(uni_embeddings.shape[1], 150)
    uni_embeddings = uni_embeddings[:, :min_uni]
    scvi_embeddings = scvi_embeddings[:, :min(scvi_embeddings.shape[1], 128)]
    rctd_embeddings = rctd_embeddings[:, :min(rctd_embeddings.shape[1], 25)]
    print(f"  - Final UNI: {uni_embeddings.shape}")
    print(f"  - Final scVI: {scvi_embeddings.shape}")
    print(f"  - Final RCTD: {rctd_embeddings.shape}\n")
    
    # Save concatenated files
    print("Saving concatenated files...")
    
    np.save(OUTPUT_DIR / "geo_uni_150d.npy", uni_embeddings.astype(np.float32))
    print(f"  ✓ Saved: geo_uni_150d.npy")
    
    np.save(OUTPUT_DIR / "geo_scvi_128d.npy", scvi_embeddings.astype(np.float32))
    print(f"  ✓ Saved: geo_scvi_128d.npy")
    
    np.save(OUTPUT_DIR / "geo_rctd_25d.npy", rctd_embeddings.astype(np.float32))
    print(f"  ✓ Saved: geo_rctd_25d.npy")
    
    np.save(OUTPUT_DIR / "geo_labels.npy", labels.astype(np.int32))
    print(f"  ✓ Saved: geo_labels.npy")
    
    barcodes_df = pd.DataFrame(barcodes, columns=['barcode'])
    barcodes_df.to_csv(OUTPUT_DIR / "geo_barcodes.csv", index=False)
    print(f"  ✓ Saved: geo_barcodes.csv")
    
    print(f"\n{'='*80}")
    print("✓ GEO DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext: Update config/geo_example.yaml with these paths:")
    print(f"  embeddings.UNI.file_path: {OUTPUT_DIR}/geo_uni_150d.npy")
    print(f"  embeddings.scVI.file_path: {OUTPUT_DIR}/geo_scvi_128d.npy")
    print(f"  embeddings.RCTD.file_path: {OUTPUT_DIR}/geo_rctd_25d.npy")
    print(f"  labels_metadata.labels_path: {OUTPUT_DIR}/geo_labels.npy")
    print(f"  labels_metadata.barcode_path: {OUTPUT_DIR}/geo_barcodes.csv")


if __name__ == "__main__":
    try:
        prepare_geo_data()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
