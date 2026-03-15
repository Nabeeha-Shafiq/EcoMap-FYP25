#!/usr/bin/env python3
"""
Helper script to convert NPY embedding files to CSV format.

This script converts existing NPY embedding files to CSV format where:
- First column: barcode
- Subsequent columns: embedding_1, embedding_2, ..., embedding_N

Usage:
    python scripts/convert_npy_to_csv.py \
        --embeddings /path/to/embeddings.npy \
        --barcodes /path/to/barcodes.csv \
        --output /path/to/output.csv \
        --modality-name "image_encoder"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def convert_npy_to_csv(embeddings_path, barcodes_path, output_path, modality_name):
    """Convert NPY embedding file to CSV with barcode column."""
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Load barcodes
    barcodes_df = pd.read_csv(barcodes_path)
    barcodes = barcodes_df['barcode'].values
    print(f"✓ Loaded barcodes: {len(barcodes)}")
    
    # Validate alignment
    if len(barcodes) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(barcodes)} barcodes vs {embeddings.shape[0]} embeddings")
    
    # Create output DataFrame
    n_dims = embeddings.shape[1]
    columns = ['barcode'] + [f'embedding_{i+1}' for i in range(n_dims)]
    
    data = np.column_stack([barcodes, embeddings])
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to CSV: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Dimensions per embedding: {n_dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY embeddings to CSV format")
    parser.add_argument("--embeddings", required=True, help="Path to NPY embeddings file")
    parser.add_argument("--barcodes", required=True, help="Path to barcodes CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--modality-name", default="embedding", help="Modality name (for column naming)")
    
    args = parser.parse_args()
    convert_npy_to_csv(args.embeddings, args.barcodes, args.output, args.modality_name)
