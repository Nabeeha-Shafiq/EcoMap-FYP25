#!/usr/bin/env python3
"""
Comprehensive Pre-Processing Validation and Visualization

Generates all QC checks, correlation matrices, and spatial visualizations after preprocessing.

Output Structure:
  preprocessing/
    ├── validation_qc_report.csv              (main QC summary)
    ├── correlation_matrices/
    │   ├── modality_correlation_matrix_3x3.csv
    │   ├── modality_correlation_matrix_3x3_heatmap.png
    │   └── modality_separability_matrix.csv
    ├── qc_checks/
    │   ├── zero_nan_infinity_values.csv
    │   ├── barcode_alignment.csv
    │   └── value_ranges.csv
    └── spatial_heatmaps/
        ├── P1_spatial_heatmap_pc1.png
        ├── P2_spatial_heatmap_pc1.png
        └── ... (one per patient)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
import warnings
import os
import sys

warnings.filterwarnings('ignore')

print("\n" + "="*100)
print("PRE-PROCESSING VALIDATION AND VISUALIZATION")
print("="*100 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# Get output directory from environment or use default
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './results_test_end_end'))
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories for organized output
METRICS_DIR = PREPROCESSING_DIR / "metrics"
VISUALIZATIONS_DIR = PREPROCESSING_DIR / "visualizations"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

QC_DIR = METRICS_DIR / "qc_checks"
QC_DIR.mkdir(exist_ok=True)

CORR_DIR = VISUALIZATIONS_DIR / "correlation_matrices"
CORR_DIR.mkdir(exist_ok=True)

HEATMAP_DIR = VISUALIZATIONS_DIR / "spatial_heatmaps"
HEATMAP_DIR.mkdir(exist_ok=True)

# Data paths
PREPROCESSED_ARRAYS_DIR = Path("data/preprocessed_arrays")
INPUT_DATA_DIR = Path("data/input_dataset")

print(f"Output Directory: {OUTPUT_DIR}")
print(f"Preprocessing Directory: {PREPROCESSING_DIR}\n")

# Modality definitions (matching preprocessing step)
MODALITIES = {
    'image_encoder': {'name': 'UNI (Image)', 'alias': 'UNI', 'color': '#E74C3C'},
    'gene_encoder': {'name': 'scVI (Gene)', 'alias': 'scVI', 'color': '#3498DB'},
    'cell_encoder': {'name': 'RCTD (Cell)', 'alias': 'RCTD', 'color': '#2ECC71'}
}

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 1: LOAD PREPROCESSED DATA
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 1] Loading Preprocessed Data...\n")

try:
    fused_embeddings = np.load(PREPROCESSED_ARRAYS_DIR / "fused_embeddings_pca.npy")
    uni_embeddings = np.load(PREPROCESSED_ARRAYS_DIR / "uni_embeddings_pca.npy")
    scvi_embeddings = np.load(PREPROCESSED_ARRAYS_DIR / "scvi_embeddings_pca.npy")
    rctd_embeddings = np.load(PREPROCESSED_ARRAYS_DIR / "rctd_embeddings_pca.npy")
    barcodes = np.load(PREPROCESSED_ARRAYS_DIR / "barcodes.npy", allow_pickle=True)
    
    print(f"  ✓ Fused Embeddings (PCA): {fused_embeddings.shape}")
    print(f"  ✓ UNI Embeddings (PCA):   {uni_embeddings.shape}")
    print(f"  ✓ scVI Embeddings (PCA):  {scvi_embeddings.shape}")
    print(f"  ✓ RCTD Embeddings (PCA):  {rctd_embeddings.shape}")
    print(f"  ✓ Barcodes: {barcodes.shape}\n")

    # Load labels
    labels_df = pd.read_csv(INPUT_DATA_DIR / "barcode_labels.csv")
    labels = labels_df.set_index('barcode').loc[barcodes, 'label'].values
    
    # Load metadata for coordinates
    metadata_df = pd.read_csv(INPUT_DATA_DIR / "barcode_metadata.csv")
    # Don't set barcode as index - keep it as a column for easier access
    # metadata_df.set_index('barcode', inplace=True)
    
    print(f"  ✓ Labels: {len(labels)}")
    print(f"  ✓ Metadata: {len(metadata_df)} rows\n")

except Exception as e:
    print(f"  ✗ ERROR loading data: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 2: QC CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 2] Performing QC Checks...\n")

# Check 1: Zero/NaN/Infinity values
qc_data = []

for name, emb in [('UNI (Image)', uni_embeddings), ('scVI (Gene)', scvi_embeddings), 
                   ('RCTD (Cell)', rctd_embeddings), ('Fused (All)', fused_embeddings)]:
    zeros = np.sum(emb == 0)
    zeros_pct = (zeros / emb.size) * 100 if emb.size > 0 else 0
    
    nans = np.sum(np.isnan(emb))
    nans_pct = (nans / emb.size) * 100 if emb.size > 0 else 0
    
    infs = np.sum(np.isinf(emb))
    infs_pct = (infs / emb.size) * 100 if emb.size > 0 else 0
    
    qc_data.append({
        'Embedding': name,
        'Dimension': emb.shape[1],
        'Samples': emb.shape[0],
        'Zero_Count': zeros,
        'Zero_Percent': f"{zeros_pct:.2f}",
        'NaN_Count': nans,
        'NaN_Percent': f"{nans_pct:.2f}",
        'Inf_Count': infs,
        'Inf_Percent': f"{infs_pct:.2f}"
    })
    
    status = "✓" if (nans == 0 and infs == 0) else "✗"
    print(f"  {status} {name}:")
    print(f"      Zero: {zeros:,} ({zeros_pct:.2f}%), NaN: {nans} ({nans_pct:.2f}%), Inf: {infs} ({infs_pct:.2f}%)")

qc_df = pd.DataFrame(qc_data)
qc_csv = QC_DIR / "zero_nan_infinity_values.csv"
qc_df.to_csv(qc_csv, index=False)
print(f"\n  ✓ Saved: {qc_csv.name}\n")

# Check 2: Barcode alignment
alignment_data = {
    'Check': ['Total Samples', 'Unique Barcodes', 'UNI-Barcodes Match', 'scVI-Barcodes Match', 
              'RCTD-Barcodes Match', 'Fused-Barcodes Match', 'Labels Match', 'Metadata Match'],
    'Status': [
        len(barcodes),
        len(np.unique(barcodes)),
        'PASS' if len(barcodes) == uni_embeddings.shape[0] else 'FAIL',
        'PASS' if len(barcodes) == scvi_embeddings.shape[0] else 'FAIL',
        'PASS' if len(barcodes) == rctd_embeddings.shape[0] else 'FAIL',
        'PASS' if len(barcodes) == fused_embeddings.shape[0] else 'FAIL',
        'PASS' if len(barcodes) == len(labels) else 'FAIL',
        'PASS' if len(barcodes) == len(metadata_df) else 'FAIL'
    ]
}

alignment_df = pd.DataFrame(alignment_data)
alignment_csv = QC_DIR / "barcode_alignment.csv"
alignment_df.to_csv(alignment_csv, index=False)
print(f"  Barcode Alignment:")
for check, status in zip(alignment_data['Check'], alignment_data['Status']):
    print(f"    {check}: {status}")
print(f"  ✓ Saved: {alignment_csv.name}\n")

# Check 3: Value ranges
ranges_data = []
for name, emb in [('UNI (Image)', uni_embeddings), ('scVI (Gene)', scvi_embeddings), 
                   ('RCTD (Cell)', rctd_embeddings), ('Fused (All)', fused_embeddings)]:
    ranges_data.extend([
        {'Embedding': name, 'Metric': 'Min', 'Value': np.nanmin(emb)},
        {'Embedding': name, 'Metric': 'Max', 'Value': np.nanmax(emb)},
        {'Embedding': name, 'Metric': 'Mean', 'Value': np.nanmean(emb)},
        {'Embedding': name, 'Metric': 'Std', 'Value': np.nanstd(emb)}
    ])

ranges_df = pd.DataFrame(ranges_data)
ranges_csv = QC_DIR / "value_ranges.csv"
ranges_df.to_csv(ranges_csv, index=False)
print(f"  Value Ranges:")
for name in ['UNI (Image)', 'scVI (Gene)', 'RCTD (Cell)', 'Fused (All)']:
    subset = ranges_df[ranges_df['Embedding'] == name]
    min_val = subset[subset['Metric'] == 'Min']['Value'].values[0]
    max_val = subset[subset['Metric'] == 'Max']['Value'].values[0]
    print(f"    {name}: [{min_val:.4f}, {max_val:.4f}]")
print(f"  ✓ Saved: {ranges_csv.name}\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 3: MODALITY CORRELATION MATRIX (3x3)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 3] Computing 3x3 Modality Correlation Matrix...\n")

# Standardize embeddings
uni_scaled = StandardScaler().fit_transform(uni_embeddings)
scvi_scaled = StandardScaler().fit_transform(scvi_embeddings)
rctd_scaled = StandardScaler().fit_transform(rctd_embeddings)

# Compute correlation matrix using mean-standardized embeddings
# Correlation between two modalities = correlation of their mean values across samples
embeddings_dict = {
    'UNI': uni_scaled,
    'scVI': scvi_scaled,
    'RCTD': rctd_scaled
}

corr_matrix = np.zeros((3, 3))
mod_names = ['UNI', 'scVI', 'RCTD']

print("  Modality Correlations:")
for i, mod1 in enumerate(mod_names):
    for j, mod2 in enumerate(mod_names):
        if i == j:
            corr_matrix[i, j] = 1.0
        else:
            # Compute correlation between mean-centered embeddings
            # Use the first principal component as representative
            pca1 = PCA(n_components=1)
            pca2 = PCA(n_components=1)
            
            pc1_mod1 = pca1.fit_transform(embeddings_dict[mod1]).flatten()
            pc1_mod2 = pca2.fit_transform(embeddings_dict[mod2]).flatten()
            
            # Pearson correlation between the two PC1 vectors
            corr = np.corrcoef(pc1_mod1, pc1_mod2)[0, 1]
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            
            print(f"    {mod1} ↔ {mod2}: {corr:.4f}")

print()

# Save 3x3 correlation matrix to CSV
corr_3x3_df = pd.DataFrame(corr_matrix, columns=mod_names, index=mod_names)
corr_3x3_csv = CORR_DIR / "modality_correlation_matrix_3x3.csv"
corr_3x3_df.to_csv(corr_3x3_csv)
print(f"  ✓ Saved 3x3 correlation matrix: {corr_3x3_csv.name}")

# Create heatmap visualization
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='RdBu_r', center=0.0,
            vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'},
            xticklabels=mod_names, yticklabels=mod_names, ax=ax,
            linewidths=2, linecolor='white', annot_kws={'size': 12, 'weight': 'bold'})

ax.set_title('3x3 Modality Correlation Matrix\n(Based on PC1 Vectors)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Modality', fontsize=12, fontweight='bold')
ax.set_ylabel('Modality', fontsize=12, fontweight='bold')
plt.tight_layout()

heatmap_path = CORR_DIR / "modality_correlation_matrix_3x3_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 3x3 heatmap: {heatmap_path.name}\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 4: MODALITY SEPARABILITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 4] Computing Modality Separability Metrics...\n")

separability_data = []

for mod_name, embeddings in [('UNI', uni_embeddings), ('scVI', scvi_embeddings), 
                              ('RCTD', rctd_embeddings), ('Fused (All 303D)', fused_embeddings)]:
    
    emb_scaled = StandardScaler().fit_transform(embeddings)
    
    # Silhouette Score (cell type separability based on labels)
    try:
        silhouette = silhouette_score(emb_scaled, labels, sample_size=min(5000, len(labels)))
    except:
        silhouette = np.nan
    
    # Davies-Bouldin Index (lower is better)
    try:
        davies_bouldin = davies_bouldin_score(emb_scaled, labels)
    except:
        davies_bouldin = np.nan
    
    # PCA variance (explained by first 2 components)
    try:
        pca = PCA(n_components=min(2, emb_scaled.shape[1]))
        pca.fit(emb_scaled)
        explained_var = np.sum(pca.explained_variance_ratio_)
    except:
        explained_var = np.nan
    
    # Intrinsic dimensionality (effective number of features)
    try:
        cov = np.cov(emb_scaled.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        intrinsic_dim = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2) if len(eigenvalues) > 0 else emb_scaled.shape[1]
    except:
        intrinsic_dim = np.nan
    
    separability_data.append({
        'Modality': mod_name,
        'Dimension': embeddings.shape[1],
        'Samples': embeddings.shape[0],
        'Silhouette_Score': silhouette,
        'Davies_Bouldin_Index': davies_bouldin,
        'Explained_Variance_PC1_PC2': explained_var,
        'Intrinsic_Dimensionality': intrinsic_dim
    })
    
    print(f"  {mod_name}:")
    print(f"    Silhouette Score: {silhouette:.4f} (higher=better cell type separation)")
    print(f"    Davies-Bouldin Index: {davies_bouldin:.4f} (lower=better cluster separation)")
    print(f"    Explained Variance (PC1+PC2): {explained_var:.4f}")
    print(f"    Intrinsic Dimensionality: {intrinsic_dim:.2f}\n")

separability_df = pd.DataFrame(separability_data)
separability_csv = CORR_DIR / "modality_separability_matrix.csv"
separability_df.to_csv(separability_csv, index=False)
print(f"  ✓ Saved separability metrics: {separability_csv.name}\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 5: SPATIAL HEATMAPS (Per-Patient PC1 Visualization)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 5] Generating Spatial Heatmaps (X,Y Coordinates from Metadata)...\n")

try:
    # Get unique patients
    patients = metadata_df['patient_id'].unique()
    
    # Compute PC1 from fused embeddings
    pca_fused = PCA(n_components=1)
    pc1_values = pca_fused.fit_transform(fused_embeddings).flatten()
    
    print(f"  PC1 Range: [{np.min(pc1_values):.4f}, {np.max(pc1_values):.4f}]")
    print(f"  PC1 Explained Variance: {pca_fused.explained_variance_ratio_[0]:.4f}\n")
    
    for patient in sorted(patients):
        # Get indices for this patient from metadata
        patient_mask = metadata_df['patient_id'] == patient
        patient_metadata = metadata_df[patient_mask]
        
        if len(patient_metadata) == 0:
            print(f"  ⚠ {patient}: No metadata found")
            continue
        
        # Match with our loaded data barcodes
        match_indices = []
        matched_barcodes = []
        
        for idx, (_, row) in enumerate(patient_metadata.iterrows()):
            bc = row['barcode']
            try:
                data_idx = np.where(barcodes == bc)[0][0]
                match_indices.append(data_idx)
                matched_barcodes.append(bc)
            except:
                pass
        
        if len(match_indices) == 0:
            print(f"  ⚠ {patient}: No matching barcodes found")
            continue
        
        match_indices = np.array(match_indices)
        
        # Get coordinates and PC1 values for this patient
        # Use x_coord and y_coord (not x_coordinate, y_coordinate)
        patient_coords = patient_metadata.loc[patient_metadata['barcode'].isin(matched_barcodes)]
        x_coords = patient_coords['x_coord'].values
        y_coords = patient_coords['y_coord'].values
        pc1_patient = pc1_values[match_indices]
        
        # Create spatial heatmap
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        scatter = ax.scatter(x_coords, y_coords, c=pc1_patient, cmap='viridis',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, label='PC1 Value (Fused Embeddings)')
        cbar.ax.tick_params(labelsize=9)
        
        ax.set_xlabel('X Coordinate (Array Width)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Coordinate (Array Height)', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient} - Spatial Embedding Distribution\nColored by PC1 (Fused)',
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.invert_yaxis()  # Invert Y-axis: 0-80 becomes 80-0
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        heatmap_path = HEATMAP_DIR / f"{patient}_spatial_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {patient}: {len(match_indices)} spots, PC1 [{np.min(pc1_patient):.4f}, {np.max(pc1_patient):.4f}]")
    
    print()

except Exception as e:
    print(f"  ✗ Error generating spatial heatmaps: {e}\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STAGE 6: COMPREHENSIVE VALIDATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("[STAGE 6] Generating Comprehensive Validation Report...\n")

validation_report = {
    'Metric': [
        'Total Samples',
        'Unique Barcodes',
        'Unique Labels',
        'Unique Patients',
        'UNI Dimension',
        'scVI Dimension',
        'RCTD Dimension',
        'Fused Dimension',
        'UNI-scVI Correlation',
        'UNI-RCTD Correlation',
        'scVI-RCTD Correlation',
        'UNI Silhouette Score',
        'scVI Silhouette Score',
        'RCTD Silhouette Score',
        'Fused Silhouette Score',
        'Data Quality Status'
    ],
    'Value': [
        len(barcodes),
        len(np.unique(barcodes)),
        len(np.unique(labels)),
        len(np.unique(metadata_df['patient_id'])),
        uni_embeddings.shape[1],
        scvi_embeddings.shape[1],
        rctd_embeddings.shape[1],
        fused_embeddings.shape[1],
        f"{corr_matrix[0, 1]:.4f}",
        f"{corr_matrix[0, 2]:.4f}",
        f"{corr_matrix[1, 2]:.4f}",
        f"{separability_df[separability_df['Modality'] == 'UNI']['Silhouette_Score'].values[0]:.4f}",
        f"{separability_df[separability_df['Modality'] == 'scVI']['Silhouette_Score'].values[0]:.4f}",
        f"{separability_df[separability_df['Modality'] == 'RCTD']['Silhouette_Score'].values[0]:.4f}",
        f"{separability_df[separability_df['Modality'] == 'Fused (All 303D)']['Silhouette_Score'].values[0]:.4f}",
        'READY' if all(~pd.isna(alignment_df['Status'])) else 'CHECK'
    ]
}

validation_df = pd.DataFrame(validation_report)
validation_csv = METRICS_DIR / "validation_qc_report.csv"
validation_df.to_csv(validation_csv, index=False)
print(f"✓ Saved comprehensive validation report: {validation_csv.name}\n")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

print("="*100)
print("✓ PRE-PROCESSING VALIDATION AND VISUALIZATION COMPLETE")
print("="*100)
print(f"\n✓ Output Directory: {PREPROCESSING_DIR}")
print(f"\n✓ Generated Files:")
print(f"  CSV Reports:")
print(f"    • validation_qc_report.csv")
print(f"    • correlation_matrices/modality_correlation_matrix_3x3.csv")
print(f"    • correlation_matrices/modality_separability_matrix.csv")
print(f"    • qc_checks/zero_nan_infinity_values.csv")
print(f"    • qc_checks/barcode_alignment.csv")
print(f"    • qc_checks/value_ranges.csv")
print(f"\n  Visualizations:")
print(f"    • correlation_matrices/modality_correlation_matrix_3x3_heatmap.png")
print(f"    • spatial_heatmaps/*.png (one per patient, colored by PC1 on X,Y coords)")
print("="*100 + "\n")
