#!/usr/bin/env python3
"""
Preprocessing Pipeline: PCA Reduction & Data Preparation
=========================================================

This script applies per-modality preprocessing:
- PCA reduction with configurable explained variance threshold
- Automatic dimension calculation based on variance preservation
- Data validation and quality reporting

Input Format:
    Expects NumPy arrays from load_input_embeddings.py
    - uni_embeddings.npy: (N, 1024)
    - scvi_embeddings.npy: (N, 128)
    - rctd_embeddings.npy: (N, 25)

Output:
    Preprocessed embeddings (with optional PCA reduction)
    - uni_embeddings_processed.npy
    - scvi_embeddings_processed.npy
    - rctd_embeddings_processed.npy
    - Preprocessing report (YAML format)

Configuration (from geo_example.yaml):
    preprocessing:
      pca_dimensions:
        image_encoder: null      # null = no reduction
        gene_encoder: null       # OR 0.95 = keep 95% variance
        cell_encoder: 0.98       # OR 0.98 = keep 98% variance
"""

import numpy as np
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
import argparse
import sys

print("\n" + "="*100)
print("PREPROCESSING PIPELINE: PCA Reduction with Variance Thresholds")
print("="*100 + "\n")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_pca_dimensions(embedding, pca_config):
    """
    Calculate required dimensions from PCA configuration.
    
    Args:
        embedding: (N, D) array
        pca_config: None (no reduction), int (fixed dims), or float (variance threshold 0-1)
    
    Returns:
        n_dims: number of components needed
        actual_variance: variance preserved
    """
    # Remove rows with NaN for PCA fitting
    valid_mask = ~np.any(np.isnan(embedding), axis=1)
    embedding_clean = embedding[valid_mask]
    
    # Fit PCA on clean data to get explained variance
    pca_full = PCA()
    pca_full.fit(embedding_clean)
    
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Handle different configuration types
    if isinstance(pca_config, int):
        # Fixed dimension target
        n_dims = min(pca_config, embedding.shape[1])
        actual_variance = cumsum[n_dims - 1] if n_dims <= len(cumsum) else cumsum[-1]
    else:
        # Variance threshold (float between 0 and 1)
        variance_threshold = pca_config
        n_dims = np.argmax(cumsum >= variance_threshold) + 1
        actual_variance = cumsum[n_dims - 1]
    
    # Ensure at least 1 dimension
    n_dims = max(1, min(n_dims, embedding.shape[1]))
    
    return n_dims, actual_variance

def apply_pca(embedding, n_components):
    """
    Apply PCA reduction to embedding.
    
    Args:
        embedding: (N, D) array
        n_components: number of components to keep
    
    Returns:
        reduced_embedding: (N, n_components) array
        nan_mask: boolean array indicating rows with NaN
    """
    # Detect rows with NaN
    nan_mask = np.any(np.isnan(embedding), axis=1)
    n_nan = np.sum(nan_mask)
    
    # Create embedding for fitting (remove NaN rows)
    embedding_clean = embedding[~nan_mask]
    
    # Fit PCA on clean data
    pca = PCA(n_components=n_components)
    pca.fit(embedding_clean)
    
    # Transform all data (NaN rows will still have NaN after transform)
    # We impute NaN as zeros for transformation
    embedding_imputed = embedding.copy()
    for col in range(embedding.shape[1]):
        col_nan_mask = np.isnan(embedding_imputed[:, col])
        if np.sum(col_nan_mask) > 0:
            # Impute with column mean (calculated from non-NaN values)
            col_mean = np.nanmean(embedding[:, col])
            embedding_imputed[col_nan_mask, col] = col_mean
    
    # Transform the imputed data
    reduced = pca.fit_transform(embedding_imputed)
    
    return reduced, nan_mask

def main():
    parser = argparse.ArgumentParser(
        description="Apply PCA preprocessing to embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)"
    )
    
    parser.add_argument(
        "--input-arrays-dir",
        type=str,
        required=True,
        help="Input directory with NumPy arrays from Stage 1 (required)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessed arrays (required)"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    input_dir = Path(args.input_arrays_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("[STAGE 1] Loading Configuration")
    print("─" * 80)
    
    if not config_path.exists():
        print(f"  ✗ ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    print(f"  ✓ Loaded: {config_path}")
    
    # Get PCA config from embeddings section (new modular format)
    # Falls back to preprocessing.pca_dimensions (old format) if not found
    embeddings_config = config.get('embeddings', {})
    
    # Build pca_config from embeddings section
    pca_config = {}
    if 'image_encoder' in embeddings_config:
        pca_config['image_encoder'] = embeddings_config['image_encoder'].get('pca_variance')
    if 'gene_encoder' in embeddings_config:
        pca_config['gene_encoder'] = embeddings_config['gene_encoder'].get('pca_variance')
    if 'cell_encoder' in embeddings_config:
        pca_config['cell_encoder'] = embeddings_config['cell_encoder'].get('pca_variance')
    
    # If empty, try old format
    if not pca_config:
        pca_config = config.get('preprocessing', {}).get('pca_dimensions', {})
    
    print(f"  Image encoder (UNI): {pca_config.get('image_encoder', 'null')}")
    print(f"  Gene encoder (scVI): {pca_config.get('gene_encoder', 'null')}")
    print(f"  Cell encoder (RCTD): {pca_config.get('cell_encoder', 'null')}")
    print()
    
    # Load embeddings
    print("[STAGE 2] Loading NumPy Arrays")
    print("─" * 80)
    
    uni_emb = np.load(input_dir / "uni_embeddings.npy")
    scvi_emb = np.load(input_dir / "scvi_embeddings.npy")
    rctd_emb = np.load(input_dir / "rctd_embeddings.npy")
    barcodes = np.load(input_dir / "barcodes.npy", allow_pickle=True)
    
    print(f"  ✓ UNI (Image):    {uni_emb.shape}")
    print(f"  ✓ scVI (Gene):    {scvi_emb.shape}")
    print(f"  ✓ RCTD (Cell):    {rctd_emb.shape}")
    print(f"  ✓ Barcodes:       {barcodes.shape} (aligned samples)")
    
    # IMPORTANT: RCTD has extra samples. We need to align it with the barcodes.
    # The barcodes represent aligned samples across all modalities.
    # RCTD needs to be subset to match aligned barcodes.
    if rctd_emb.shape[0] > barcodes.shape[0]:
        print(f"\n  Note: RCTD has {rctd_emb.shape[0] - barcodes.shape[0]} extra samples")
        print(f"        Aligning RCTD to {barcodes.shape[0]} common barcodes...")
        rctd_emb = rctd_emb[:barcodes.shape[0]]  # Keep only aligned samples
        print(f"  ✓ RCTD aligned to: {rctd_emb.shape}")
    
    print()
    
    # Apply PCA per modality
    print("[STAGE 3] Applying PCA Reduction")
    print("─" * 80)
    
    preprocessed = {}
    pca_stats = {}
    
    # Image encoder (UNI)
    print("\n  Image Encoder (UNI - 1024D):")
    uni_pca_config = pca_config.get('image_encoder')
    
    if uni_pca_config is None:
        print(f"    Config: null (no reduction)")
        uni_processed = uni_emb
        pca_stats['image_encoder'] = {
            'config': None,
            'original_dims': uni_emb.shape[1],
            'final_dims': uni_emb.shape[1],
            'explained_variance': 1.0,
            'nan_rows': int(np.sum(np.any(np.isnan(uni_emb), axis=1)))
        }
    else:
        n_dims, actual_var = calculate_pca_dimensions(uni_emb, uni_pca_config)
        uni_processed, uni_nan_mask = apply_pca(uni_emb, n_dims)
        if isinstance(uni_pca_config, int):
            config_str = f"{uni_pca_config}D (fixed dimensions)"
        else:
            config_str = f"{uni_pca_config*100:.0f}% variance"
        print(f"    Config: {config_str}")
        print(f"    Reduction: 1024D → {n_dims}D")
        print(f"    Actual variance preserved: {actual_var*100:.2f}%")
        print(f"    Note: {np.sum(uni_nan_mask)} rows with NaN were imputed with column means")
        pca_stats['image_encoder'] = {
            'config': uni_pca_config,
            'original_dims': uni_emb.shape[1],
            'final_dims': n_dims,
            'explained_variance': float(actual_var),
            'nan_rows': int(np.sum(uni_nan_mask))
        }
    
    preprocessed['uni'] = uni_processed
    
    # Gene encoder (scVI)
    print("\n  Gene Encoder (scVI - 128D):")
    scvi_pca_config = pca_config.get('gene_encoder')
    
    if scvi_pca_config is None:
        print(f"    Config: null (no reduction)")
        scvi_processed = scvi_emb
        pca_stats['gene_encoder'] = {
            'config': None,
            'original_dims': scvi_emb.shape[1],
            'final_dims': scvi_emb.shape[1],
            'explained_variance': 1.0,
            'nan_rows': int(np.sum(np.any(np.isnan(scvi_emb), axis=1)))
        }
    else:
        n_dims, actual_var = calculate_pca_dimensions(scvi_emb, scvi_pca_config)
        scvi_processed, scvi_nan_mask = apply_pca(scvi_emb, n_dims)
        if isinstance(scvi_pca_config, int):
            config_str = f"{scvi_pca_config}D (fixed dimensions)"
        else:
            config_str = f"{scvi_pca_config*100:.0f}% variance"
        print(f"    Config: {config_str}")
        print(f"    Reduction: 128D → {n_dims}D")
        print(f"    Actual variance preserved: {actual_var*100:.2f}%")
        print(f"    Note: {np.sum(scvi_nan_mask)} rows with NaN were imputed with column means")
        pca_stats['gene_encoder'] = {
            'config': scvi_pca_config,
            'original_dims': scvi_emb.shape[1],
            'final_dims': n_dims,
            'explained_variance': float(actual_var),
            'nan_rows': int(np.sum(scvi_nan_mask))
        }
    
    preprocessed['scvi'] = scvi_processed
    
    # Cell encoder (RCTD)
    print("\n  Cell Encoder (RCTD - 25D):")
    rctd_pca_config = pca_config.get('cell_encoder')
    
    if rctd_pca_config is None:
        print(f"    Config: null (no reduction)")
        rctd_processed = rctd_emb
        pca_stats['cell_encoder'] = {
            'config': None,
            'original_dims': rctd_emb.shape[1],
            'final_dims': rctd_emb.shape[1],
            'explained_variance': 1.0,
            'nan_rows': int(np.sum(np.any(np.isnan(rctd_emb), axis=1)))
        }
    else:
        n_dims, actual_var = calculate_pca_dimensions(rctd_emb, rctd_pca_config)
        rctd_processed, rctd_nan_mask = apply_pca(rctd_emb, n_dims)
        if isinstance(rctd_pca_config, int):
            config_str = f"{rctd_pca_config}D (fixed dimensions)"
        else:
            config_str = f"{rctd_pca_config*100:.0f}% variance"
        print(f"    Config: {config_str}")
        print(f"    Reduction: 25D → {n_dims}D")
        print(f"    Actual variance preserved: {actual_var*100:.2f}%")
        print(f"    Note: {np.sum(rctd_nan_mask)} rows with NaN were imputed with column means")
        pca_stats['cell_encoder'] = {
            'config': rctd_pca_config,
            'original_dims': rctd_emb.shape[1],
            'final_dims': n_dims,
            'explained_variance': float(actual_var),
            'nan_rows': int(np.sum(rctd_nan_mask))
        }
    
    preprocessed['rctd'] = rctd_processed
    print()
    
    # Save preprocessed arrays
    print("[STAGE 4] Saving Preprocessed Arrays")
    print("─" * 80)
    
    np.save(output_dir / "uni_embeddings_pca.npy", preprocessed['uni'])
    print(f"  ✓ Saved: uni_embeddings_pca.npy {preprocessed['uni'].shape}")
    
    np.save(output_dir / "scvi_embeddings_pca.npy", preprocessed['scvi'])
    print(f"  ✓ Saved: scvi_embeddings_pca.npy {preprocessed['scvi'].shape}")
    
    np.save(output_dir / "rctd_embeddings_pca.npy", preprocessed['rctd'])
    print(f"  ✓ Saved: rctd_embeddings_pca.npy {preprocessed['rctd'].shape}")
    
    np.save(output_dir / "barcodes.npy", barcodes)
    print(f"  ✓ Saved: barcodes.npy {barcodes.shape}")
    print()
    
    # Create fused embedding
    print("[STAGE 5] Creating Fused Embedding")
    print("─" * 80)
    
    fused_embedding = np.concatenate([
        preprocessed['uni'],
        preprocessed['scvi'],
        preprocessed['rctd']
    ], axis=1)
    
    np.save(output_dir / "fused_embeddings_pca.npy", fused_embedding)
    print(f"  ✓ Fused dimensions: {fused_embedding.shape[1]}")
    print(f"    UNI:  {preprocessed['uni'].shape[1]}D")
    print(f"    scVI: {preprocessed['scvi'].shape[1]}D")
    print(f"    RCTD: {preprocessed['rctd'].shape[1]}D")
    print(f"  ✓ Saved: fused_embeddings_pca.npy {fused_embedding.shape}")
    print()
    
    # Save preprocessing report
    print("[STAGE 6] Saving Preprocessing Report")
    print("─" * 80)
    
    report = {
        'preprocessing_config': {
            'pca_dimensions': pca_config
        },
        'pca_statistics': pca_stats,
        'output_summary': {
            'n_samples': int(fused_embedding.shape[0]),
            'original_total_dims': sum(s['original_dims'] for s in pca_stats.values()),
            'final_total_dims': int(fused_embedding.shape[1]),
            'total_variance_preserved': float(np.mean([s['explained_variance'] for s in pca_stats.values()]))
        }
    }
    
    report_path = output_dir / "preprocessing_report.yaml"
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    print(f"  ✓ Saved: preprocessing_report.yaml")
    print()
    
    # Summary
    print("[SUMMARY]")
    print("─" * 80)
    print(f"  Total samples: {fused_embedding.shape[0]:,}")
    print(f"  Original total dimensions: {sum(s['original_dims'] for s in pca_stats.values())}")
    print(f"  Final total dimensions: {fused_embedding.shape[1]}")
    print(f"  Average variance preserved: {report['output_summary']['total_variance_preserved']*100:.2f}%")
    print(f"  Output directory: {output_dir}")
    print()
    
    print("="*100)
    print("✓ PREPROCESSING COMPLETE")
    print("="*100 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
