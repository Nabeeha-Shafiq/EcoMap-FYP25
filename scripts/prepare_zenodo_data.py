#!/usr/bin/env python3
"""
Helper script to prepare Zenodo dataset for modular pipeline testing.

Creates concatenated embeddings from individual NP files.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
FYP2_DIR = Path("/home/hp/Desktop/dataset_folder/FYP-2")
ZENODO_DATA_DIR = FYP2_DIR / "embeddings-for-training-ZENODO"
GROUND_TRUTH_DIR = ZENODO_DATA_DIR / "ground-truth"
OUTPUT_DIR = FYP2_DIR / "prepared_for_modular_pipeline"
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_zenodo_data():
    """Prepare Zenodo data for modular pipeline."""
    print("="*80)
    print("PREPARING ZENODO DATA FOR MODULAR PIPELINE")
    print("="*80 + "\n")
    
    # UNI embeddings - multiple folders per sample
    print("Concatenating UNI embeddings...")
    uni_folder = ZENODO_DATA_DIR / "UNI"
    uni_files = sorted(list(uni_folder.rglob("*embeddings.npy")))
    print(f"Found {len(uni_files)} UNI files:")
    uni_data = []
    for f in uni_files:
        print(f"  - {f.relative_to(ZENODO_DATA_DIR)}")
        uni_data.append(np.load(f))
    uni_embeddings = np.vstack(uni_data)
    print(f"  → Combined shape: {uni_embeddings.shape}\n")
    
    # scVI embeddings
    print("Concatenating scVI embeddings...")
    scvi_folder = ZENODO_DATA_DIR / "scVI"
    scvi_files = sorted(list(scvi_folder.glob("*embeddings.npy")))
    print(f"Found {len(scvi_files)} scVI files:")
    scvi_data = []
    for f in scvi_files:
        print(f"  - {f.name}")
        scvi_data.append(np.load(f))
    scvi_embeddings = np.vstack(scvi_data)
    print(f"  → Combined shape: {scvi_embeddings.shape}\n")
    
    # RCTD embeddings (cell composition)
    print("Concatenating RCTD embeddings...")
    rctd_folder = ZENODO_DATA_DIR / "Cell Composition"
    rctd_file = rctd_folder / "all_slides_combined.csv"
    df_rctd = pd.read_csv(rctd_file)
    # Remove barcode and ID columns
    numeric_cols = [c for c in df_rctd.columns if c not in ['barcode', 'ID', 'slide']]
    rctd_embeddings = df_rctd[numeric_cols].values
    print(f"  - Loaded from {rctd_file.name}")
    print(f"  → Shape: {rctd_embeddings.shape}\n")
    
    # Use UNI as source of truth for sample count
    n_samples = uni_embeddings.shape[0]
    print(f"Using UNI embedding count as truth: {n_samples} samples")
    
    # Trim all to same size
    if scvi_embeddings.shape[0] < n_samples:
        print(f"  ⚠ scVI has fewer samples ({scvi_embeddings.shape[0]}), using available")
        n_samples = min(scvi_embeddings.shape[0], n_samples)
    if rctd_embeddings.shape[0] < n_samples:
        print(f"  ⚠ RCTD has fewer samples ({rctd_embeddings.shape[0]}), using available")
        n_samples = min(rctd_embeddings.shape[0], n_samples)
    
    print(f"Final sample count: {n_samples}\n")
    
    # Trim
    uni_embeddings = uni_embeddings[:n_samples]
    scvi_embeddings = scvi_embeddings[:n_samples]
    rctd_embeddings = rctd_embeddings[:n_samples]
    
    # Trim dimensions
    print("Trimming dimensions to spec...")
    uni_embeddings = uni_embeddings[:, :150]
    scvi_embeddings = scvi_embeddings[:, :32]
    rctd_embeddings = rctd_embeddings[:, :15]
    print(f"  - UNI: {uni_embeddings.shape}")
    print(f"  - scVI: {scvi_embeddings.shape}")
    print(f"  - RCTD: {rctd_embeddings.shape}\n")
    
    # Create synthetic barcodes and labels if needed
    print("Creating synthetic barcodes and labels...")
    barcodes = np.array([f"ZENODO_{i:08d}" for i in range(n_samples)])
    # Random labels for test (since we don't have ground truth readily available)
    labels = np.random.randint(0, 5, n_samples).astype(np.int32)
    print(f"  ✓ Barcodes: {barcodes.shape}\n")
    print(f"  ✓ Labels: {labels.shape}\n")
    
    # Save
    print("Saving concatenated files...")
    np.save(OUTPUT_DIR / "zenodo_uni_150d.npy", uni_embeddings.astype(np.float32))
    print(f"  ✓ Saved: zenodo_uni_150d.npy")
    
    np.save(OUTPUT_DIR / "zenodo_scvi_32d.npy", scvi_embeddings.astype(np.float32))
    print(f"  ✓ Saved: zenodo_scvi_32d.npy")
    
    np.save(OUTPUT_DIR / "zenodo_rctd_15d.npy", rctd_embeddings.astype(np.float32))
    print(f"  ✓ Saved: zenodo_rctd_15d.npy")
    
    np.save(OUTPUT_DIR / "zenodo_labels.npy", labels)
    print(f"  ✓ Saved: zenodo_labels.npy")
    
    barcodes_df = pd.DataFrame(barcodes, columns=['barcode'])
    barcodes_df.to_csv(OUTPUT_DIR / "zenodo_barcodes.csv", index=False)
    print(f"  ✓ Saved: zenodo_barcodes.csv")
    
    print(f"\n{'='*80}")
    print("✓ ZENODO DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext: Update config/zenodo_example.yaml with these paths:")
    print(f"  embeddings.UNI.file_path: {OUTPUT_DIR}/zenodo_uni_150d.npy")
    print(f"  embeddings.scVI.file_path: {OUTPUT_DIR}/zenodo_scvi_32d.npy")
    print(f"  embeddings.RCTD.file_path: {OUTPUT_DIR}/zenodo_rctd_15d.npy")
    print(f"  labels_metadata.labels_path: {OUTPUT_DIR}/zenodo_labels.npy")
    print(f"  labels_metadata.barcode_path: {OUTPUT_DIR}/zenodo_barcodes.csv")


if __name__ == "__main__":
    try:
        prepare_zenodo_data()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
