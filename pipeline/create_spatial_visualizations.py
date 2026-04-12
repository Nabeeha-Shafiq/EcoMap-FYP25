#!/usr/bin/env python3
"""
Spatial Localization Visualizations for Modular Pipeline
=========================================================

Creates 4 visualization types per patient:
1. Spatial Ecotype Maps (3-panel: Ground Truth | Predictions | Accuracy)
2. Confidence Heatmaps (2-panel: Confidence Map | Uncertainty Regions)
3. Neighborhood Analysis (6-panel: Per-ecotype Moran's I + summary)
4. 3D Tissue Landscape (Interactive HTML: X,Y tissue × Z cell composition)

Author: FYP-Modular Pipeline
Date: March 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import argparse
import sys

# For 3D visualization
import plotly.graph_objects as go

# For spatial statistics
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (with command-line override support)
# ============================================================================

# Default paths (DEPRECATED - use command-line arguments or config instead)
# These constants are kept for reference only and should NOT be used as fallbacks
# ALL paths must be explicitly provided via arguments or config file
DEFAULT_PREDICTIONS_FILE = None  # Not used - must be provided via --predictions
DEFAULT_EMBEDDINGS_FILE = None  # Not used - must be provided via --embeddings
DEFAULT_LABEL_MAPPING_FILE = None  # Not used - must be provided via config or --label-mapping
DEFAULT_METADATA_FILE = None  # Not used - must be provided via config or --metadata
DEFAULT_OUTPUT_DIR = None  # Not used - must be provided via --output

# These will be set by arguments or defaults
PREDICTIONS_FILE = None
EMBEDDINGS_FILE = None
LABEL_MAPPING_FILE = None
METADATA_FILE = None
OUTPUT_DIR = None

# Ecotype colors (from reference code)
ECOTYPE_COLORS = {
    'Fibrotic': '#E74C3C',              # Red
    'Immunosuppressive': '#3498DB',     # Blue
    'Invasive_Border': '#F39C12',       # Orange
    'Metabolic': '#2ECC71',             # Green
    'Normal_Adjacent': '#9B59B6'        # Purple
}

LABEL_TO_ECOTYPE = {
    0: 'Fibrotic',
    1: 'Immunosuppressive',
    2: 'Invasive_Border',
    3: 'Metabolic',
    4: 'Normal_Adjacent'
}

# Note: OUTPUT_DIR will be initialized in main() after argument parsing
# Note: PATIENTS list is NO LONGER HARDCODED - extracted dynamically from data


# ============================================================================
# CORE VISUALIZATION CLASS
# ============================================================================

class SpatialVisualizationPipeline:
    """Comprehensive spatial visualization suite for ecotype analysis."""
    
    def __init__(self):
        """Initialize visualization pipeline."""
        self.data = None
        self.metadata = None
        print(f"✅ Spatial Visualization Pipeline initialized")
        print(f"   Output directory: {OUTPUT_DIR}")
    
    def load_data(self):
        """Load predictions, embeddings, and metadata."""
        
        print("\n" + "="*80)
        print("[STAGE 1] Loading data...")
        print("="*80)
        
        # Load predictions
        if not Path(PREDICTIONS_FILE).exists():
            raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_FILE}")
        
        predictions_df = pd.read_csv(PREDICTIONS_FILE)
        print(f"✓ Predictions loaded: {len(predictions_df)} spots")
        print(f"  Patients: {sorted(predictions_df['patient_id'].unique())}")
        
        # Load embeddings for PCA coordinates (303D fused embeddings)
        if not Path(EMBEDDINGS_FILE).exists():
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        
        embeddings_array = np.load(EMBEDDINGS_FILE)
        print(f"✓ Embeddings loaded: {embeddings_array.shape} (fused: UNI 150D + scVI 128D + RCTD 25D)")
        
        # Load label mapping
        with open(LABEL_MAPPING_FILE, 'r') as f:
            label_mapping = json.load(f)
        label_names = [label_mapping['labels'][str(i)] for i in range(5)]
        print(f"✓ Label mapping loaded: {label_names}")
        
        # Load metadata for barcode indexes
        metadata_df = pd.read_csv(METADATA_FILE)
        print(f"✓ Metadata loaded: {len(metadata_df)} spots")
        
        # Create ecotype name columns
        predictions_df['ground_truth_ecotype'] = predictions_df['ground_truth_label'].map(LABEL_TO_ECOTYPE)
        predictions_df['predicted_ecotype'] = predictions_df['predicted_label'].map(LABEL_TO_ECOTYPE)
        
        # Verify confidence column exists from training/predictions
        if 'confidence' in predictions_df.columns:
            print(f"✓ Confidence scores loaded (mean: {predictions_df['confidence'].mean():.3f}, "
                  f"std: {predictions_df['confidence'].std():.3f})")
        else:
            print(f"⚠️  Warning: Confidence column not found in predictions CSV")
            print(f"   Make sure to run training with updated train_mlp.py script")
            predictions_df['confidence'] = 0.5  # Default fallback
        
        print("\n" + "="*80)
        print("[STAGE 2] Computing PCA coordinates from embeddings (per-patient)...")
        print("="*80)
        print("NOTE: Using PCA of 303D embeddings to generate PC1, PC2 coordinates")
        print("       array_col = PC1, array_row = PC2 (embedding space coordinates)")
        print("="*80)
        
        # Create barcode to embedding index mapping
        metadata_df_sorted = metadata_df.sort_values('barcode').reset_index(drop=True)
        barcode_to_embedding_idx = {bc: idx for idx, bc in enumerate(metadata_df_sorted['barcode'])}
        
        # Initialize PCA coordinate columns
        predictions_df['array_col'] = np.nan
        predictions_df['array_row'] = np.nan
        
        # Extract unique patient IDs dynamically from data (NOT hardcoded)
        # Barcodes are formatted as: patient_id_barcode_suffix
        # Extract patient_id from predictions DataFrame directly
        unique_patients = sorted(predictions_df['patient_id'].unique())
        
        # For each patient, apply PCA to their embeddings
        for patient_id in unique_patients:
            # Get all spots for this patient
            patient_mask = predictions_df['patient_id'] == patient_id
            patient_spots = predictions_df[patient_mask].copy()
            
            if patient_spots.empty:
                print(f"  ⚠️  {patient_id}: No spots found")
                continue
            
            # Get embedding indices for this patient's barcodes
            embedding_indices = []
            valid_rows = []
            
            for idx, (df_idx, row) in enumerate(predictions_df[patient_mask].iterrows()):
                barcode = row['barcode']
                emb_idx = barcode_to_embedding_idx.get(barcode)
                
                if emb_idx is not None:
                    embedding_indices.append(emb_idx)
                    valid_rows.append(df_idx)
            
            if not embedding_indices:
                print(f"  ⚠️  {patient_id}: No valid embeddings found")
                continue
            
            # Extract patient's embeddings
            patient_embeddings = embeddings_array[embedding_indices]  # Shape: (n_spots, 303)
            
            # Apply PCA to reduce 303D embeddings to 2D
            pca = PCA(n_components=2)
            patient_pca_coords = pca.fit_transform(patient_embeddings)  # Shape: (n_spots, 2)
            
            # PC1 explained variance
            variance_explained = pca.explained_variance_ratio_.sum()
            
            # Assign PC1, PC2 coordinates to dataframe
            predictions_df.loc[valid_rows, 'array_col'] = patient_pca_coords[:, 0]  # PC1
            predictions_df.loc[valid_rows, 'array_row'] = patient_pca_coords[:, 1]  # PC2
            
            print(f"  ✓ {patient_id}: {len(embedding_indices)} spots | "
                  f"Variance explained: {variance_explained:.2%} | "
                  f"PC1 range: [{patient_pca_coords[:, 0].min():.2f}, {patient_pca_coords[:, 0].max():.2f}] | "
                  f"PC2 range: [{patient_pca_coords[:, 1].min():.2f}, {patient_pca_coords[:, 1].max():.2f}]")
        
        # Drop rows with missing coordinates
        rows_before = len(predictions_df)
        predictions_df = predictions_df.dropna(subset=['array_col', 'array_row'])
        rows_after = len(predictions_df)
        
        if rows_before > rows_after:
            print(f"\n⚠️  Dropped {rows_before - rows_after} rows with missing coordinates")
        
        self.data = predictions_df
        self.metadata = metadata_df
        
        print(f"\n✓ Data loading complete:")
        print(f"  - Total valid spots: {len(self.data)}")
        print(f"  - Embedding coordinates (array_col, array_row) assigned from PCA")
        
        return predictions_df
    
    def create_spatial_ecotype_map(self, patient_id='P1', figsize=(22, 8)):
        """
        Create 3-panel spatial ecotype map:
        Panel 1: Ground Truth | Panel 2: Predictions | Panel 3: Accuracy
        """
        
        patient_data = self.data[self.data['patient_id'] == patient_id].dropna(subset=['array_col', 'array_row'])
        
        if len(patient_data) == 0:
            print(f"  ⚠️  No valid data for {patient_id}")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # ===== PANEL 1: GROUND TRUTH =====
        ax1 = axes[0]
        for ecotype in ECOTYPE_COLORS.keys():
            mask = patient_data['ground_truth_ecotype'] == ecotype
            if mask.sum() > 0:
                ax1.scatter(
                    patient_data.loc[mask, 'array_col'],
                    patient_data.loc[mask, 'array_row'],
                    c=ECOTYPE_COLORS[ecotype],
                    label=f"{ecotype} (n={mask.sum()})",
                    s=60, alpha=0.7, edgecolors='black', linewidths=0.5
                )
        
        ax1.set_title(f'{patient_id} - Ground Truth Labels\n(Embedding Space Coordinates)', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('PC1 - 1st Principal Component (from 303D embeddings)', fontsize=11)
        ax1.set_ylabel('PC2 - 2nd Principal Component (from 303D embeddings)', fontsize=11)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # ===== PANEL 2: PREDICTIONS =====
        ax2 = axes[1]
        for ecotype in ECOTYPE_COLORS.keys():
            mask = patient_data['predicted_ecotype'] == ecotype
            if mask.sum() > 0:
                ax2.scatter(
                    patient_data.loc[mask, 'array_col'],
                    patient_data.loc[mask, 'array_row'],
                    c=ECOTYPE_COLORS[ecotype],
                    label=f"{ecotype} (n={mask.sum()})",
                    s=60, alpha=0.7, edgecolors='black', linewidths=0.5
                )
        
        overall_accuracy = (patient_data['ground_truth_ecotype'] == patient_data['predicted_ecotype']).mean() * 100
        
        ax2.set_title(f'{patient_id} - Model Predictions\n(Accuracy: {overall_accuracy:.2f}%)', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('PC1 - 1st Principal Component (from 303D embeddings)', fontsize=11)
        ax2.set_ylabel('PC2 - 2nd Principal Component (from 303D embeddings)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # ===== PANEL 3: ACCURACY MAP =====
        ax3 = axes[2]
        
        correct = patient_data['ground_truth_ecotype'] == patient_data['predicted_ecotype']
        
        # Plot CORRECT predictions in GREEN
        if correct.sum() > 0:
            ax3.scatter(
                patient_data.loc[correct, 'array_col'],
                patient_data.loc[correct, 'array_row'],
                c='#2ECC71',  # Green
                label=f"Correct (n={correct.sum()})",
                s=60, alpha=0.6, edgecolors='darkgreen', linewidths=0.5, marker='o'
            )
        
        # Plot INCORRECT predictions in RED
        if (~correct).sum() > 0:
            ax3.scatter(
                patient_data.loc[~correct, 'array_col'],
                patient_data.loc[~correct, 'array_row'],
                c='#E74C3C',  # Red
                label=f"Errors (n={(~correct).sum()})",
                s=100, alpha=0.8, edgecolors='darkred', linewidths=1, marker='X'
            )
        
        accuracy = correct.mean() * 100
        
        ax3.set_title(f'{patient_id} - Accuracy Map\n(Accuracy: {accuracy:.2f}%)', 
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('PC1 - 1st Principal Component (from 303D embeddings)', fontsize=11)
        ax3.set_ylabel('PC2 - 2nd Principal Component (from 303D embeddings)', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # Add accuracy annotation
        ax3.text(0.02, 0.98, f'Correct: {correct.sum()}\nTotal: {len(patient_data)}',
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f'{patient_id}_spatial_ecotype_map.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def create_confidence_heatmap(self, patient_id='P1', figsize=(20, 8)):
        """
        Create 2-panel confidence heatmap:
        Panel 1: Confidence Map | Panel 2: Uncertainty Regions
        """
        
        patient_data = self.data[self.data['patient_id'] == patient_id].dropna(subset=['array_col', 'array_row'])
        
        if len(patient_data) == 0 or 'confidence' not in patient_data.columns:
            print(f"  ⚠️  No confidence data for {patient_id}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Confidence heatmap
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            patient_data['array_col'],
            patient_data['array_row'],
            c=patient_data['confidence'],
            cmap='RdYlGn', s=80, alpha=0.8, edgecolors='black', linewidths=0.5, vmin=0, vmax=1
        )
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Prediction Confidence', fontsize=12, fontweight='bold')
        
        ax1.set_title(f'{patient_id} - Confidence Map\n(Embedding Space, Higher = More Certain)', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('PC1 - 1st Principal Component (from 303D embeddings)', fontsize=11)
        ax1.set_ylabel('PC2 - 2nd Principal Component (from 303D embeddings)', fontsize=11)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        mean_conf = patient_data['confidence'].mean()
        std_conf = patient_data['confidence'].std()
        ax1.text(0.02, 0.98, f'Mean: {mean_conf:.3f}\nStd: {std_conf:.3f}',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 2: Uncertainty regions
        ax2 = axes[1]
        
        high_conf = patient_data['confidence'] > 0.8
        ax2.scatter(
            patient_data.loc[high_conf, 'array_col'],
            patient_data.loc[high_conf, 'array_row'],
            c='lightgreen', label=f'High (>0.8): n={high_conf.sum()}',
            s=50, alpha=0.5, edgecolors='black', linewidths=0.5
        )
        
        med_conf = (patient_data['confidence'] >= 0.6) & (patient_data['confidence'] <= 0.8)
        ax2.scatter(
            patient_data.loc[med_conf, 'array_col'],
            patient_data.loc[med_conf, 'array_row'],
            c='yellow', label=f'Med (0.6-0.8): n={med_conf.sum()}',
            s=60, alpha=0.7, edgecolors='black', linewidths=0.5
        )
        
        low_conf = patient_data['confidence'] < 0.6
        ax2.scatter(
            patient_data.loc[low_conf, 'array_col'],
            patient_data.loc[low_conf, 'array_row'],
            c='red', label=f'Low (<0.6): n={low_conf.sum()}',
            s=100, alpha=0.9, edgecolors='black', linewidths=1, marker='s'
        )
        
        ax2.set_title(f'{patient_id} - Uncertainty Regions\n(Embedding Space, Red = Model Unsure)', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('PC1 - 1st Principal Component (from 303D embeddings)', fontsize=11)
        ax2.set_ylabel('PC2 - 2nd Principal Component (from 303D embeddings)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f'{patient_id}_confidence_heatmap.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def calculate_morans_i(self, patient_data, variable_col, k_neighbors=6):
        """Calculate Moran's I spatial autocorrelation statistic."""
        
        coords = patient_data[['array_col', 'array_row']].values
        values = patient_data[variable_col].values
        
        if len(values) < 3:
            return np.nan
        
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(values))).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        n = len(values)
        W = np.zeros((n, n))
        for i in range(n):
            for j in indices[i, 1:]:
                W[i, j] = 1
        
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        mean_val = values.mean()
        deviations = values - mean_val
        
        if deviations.std() < 1e-10:
            return np.nan
        
        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations ** 2)
        
        if denominator == 0:
            return np.nan
        
        morans_i = (n / W.sum()) * (numerator / denominator)
        
        return morans_i
    
    def create_neighborhood_analysis(self, patient_id='P1', figsize=(20, 12)):
        """
        Create 6-panel neighborhood analysis with Moran's I statistics:
        Panels 1-5: Per-ecotype analysis
        Panel 6: Summary card
        """
        
        patient_data = self.data[self.data['patient_id'] == patient_id].dropna(subset=['array_col', 'array_row']).copy()
        
        if len(patient_data) == 0:
            print(f"  ⚠️  No valid data for {patient_id}")
            return
        
        # Calculate Moran's I for overall predicted labels
        morans_i_overall = self.calculate_morans_i(patient_data, 'predicted_label', k_neighbors=6)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        ecotype_list = list(ECOTYPE_COLORS.keys())
        
        # Panels 1-5: Per-ecotype analysis
        for idx, ecotype in enumerate(ecotype_list):
            ax = axes[idx]
            
            ecotype_mask = patient_data['ground_truth_ecotype'] == ecotype
            ecotype_data = patient_data[ecotype_mask].copy()
            
            if len(ecotype_data) > 10:
                morans_i_ecotype = self.calculate_morans_i(ecotype_data, 'predicted_label', k_neighbors=6)
            else:
                morans_i_ecotype = np.nan
            
            # Plot non-target ecotypes in gray
            ax.scatter(
                patient_data.loc[~ecotype_mask, 'array_col'],
                patient_data.loc[~ecotype_mask, 'array_row'],
                c='lightgray', s=30, alpha=0.3
            )
            
            # Plot target ecotype in color
            if len(ecotype_data) > 0:
                ax.scatter(
                    ecotype_data['array_col'],
                    ecotype_data['array_row'],
                    c=ECOTYPE_COLORS[ecotype],
                    s=80, alpha=0.8, edgecolors='black', linewidths=0.5,
                    label=f'{ecotype}'
                )
            
            # Determine clustering status
            if not np.isnan(morans_i_ecotype):
                clustering_text = "Clustered" if morans_i_ecotype > 0.2 else "Random"
                title_text = f"{ecotype}\nMoran's I = {morans_i_ecotype:.3f} ({clustering_text})\nn = {len(ecotype_data)}"
            else:
                title_text = f"{ecotype}\n(Insufficient data)\nn = {len(ecotype_data)}"
            
            ax.set_title(title_text, fontsize=11, fontweight='bold')
            ax.set_xlabel('PC1 (from 303D embeddings)', fontsize=10)
            ax.set_ylabel('PC2 (from 303D embeddings)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        # Panel 6: Summary text
        ax = axes[5]
        ax.axis('off')
        
        overall_status = "CLUSTERED" if morans_i_overall > 0.2 else "RANDOM" if not np.isnan(morans_i_overall) else "N/A"
        morans_val_text = f"{morans_i_overall:.4f}" if not np.isnan(morans_i_overall) else "N/A"
        
        summary_text = f"""SPATIAL CLUSTERING SUMMARY
{patient_id}

Overall Moran's I: {morans_val_text}
Result: {overall_status}

Interpretation:
• I > 0.3: Strongly Clustered
• I 0.1-0.3: Weakly Clustered
• I < 0.1: Random/Dispersed

Biological Meaning:
• Clustered = Organized TME
• Random = Heterogeneous TME

Total Spots: {len(patient_data)}"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'{patient_id} - Neighborhood Analysis (Moran\'s I)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = OUTPUT_DIR / f'{patient_id}_neighborhood_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def create_3d_tissue_landscape(self, patient_id='P1'):
        """
        Create interactive 3D tissue landscape visualization.
        X,Y = PC1/PC2 embedding coordinates | Z = prediction confidence
        """
        
        patient_data = self.data[self.data['patient_id'] == patient_id].dropna(subset=['array_col', 'array_row']).copy()
        
        if len(patient_data) == 0:
            print(f"  ⚠️  No valid data for {patient_id}")
            return
        
        # Use confidence as Z-axis if available, else use normalized prediction label
        if 'confidence' in patient_data.columns:
            patient_data['z_value'] = patient_data['confidence']
            z_label = 'Prediction Confidence'
        else:
            patient_data['z_value'] = patient_data['predicted_label'] / 4.0
            z_label = 'Predicted Ecotype'
        
        z_min, z_max = patient_data['z_value'].min(), patient_data['z_value'].max()
        if z_max > z_min:
            patient_data['z_norm'] = (patient_data['z_value'] - z_min) / (z_max - z_min)
        else:
            patient_data['z_norm'] = 0.5
        
        fig = go.Figure()
        
        # Add trace for each ecotype
        for ecotype in ECOTYPE_COLORS.keys():
            mask = patient_data['predicted_ecotype'] == ecotype
            if mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=patient_data.loc[mask, 'array_col'],
                    y=patient_data.loc[mask, 'array_row'],
                    z=patient_data.loc[mask, 'z_norm'],
                    mode='markers',
                    name=f'{ecotype} (n={mask.sum()})',
                    marker=dict(
                        size=6,
                        color=ECOTYPE_COLORS[ecotype],
                        opacity=0.8,
                        line=dict(color='black', width=0.5),
                        sizemode='diameter'
                    ),
                    text=patient_data.loc[mask].apply(
                        lambda row: f"<b>{row['predicted_ecotype']}</b><br>"
                                  f"Confidence: {row.get('confidence', 0):.3f}<br>"
                                  f"Embedding Position: (PC1={row['array_col']:.2f}, PC2={row['array_row']:.2f})<br>"
                                  f"<i>X,Y = Embedding Space | Higher Z = Higher Confidence</i>", 
                        axis=1
                    ),
                    hoverinfo='text'
                ))
        
        fig.update_layout(
            title=f'{patient_id} - 3D Tissue Landscape<br>'
                  f'<sub>X,Y = Embedding Space (PC1, PC2 from 303D) | Z = {z_label} | Color = Ecotype</sub>',
            scene=dict(
                xaxis_title='PC1 - 1st Principal Component (from 303D embeddings)',
                yaxis_title='PC2 - 2nd Principal Component (from 303D embeddings)',
                zaxis_title=f'{z_label}',
                zaxis=dict(nticks=4, range=[0, 1]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            width=1400, height=1000, font=dict(size=12),
            hovermode='closest'
        )
        
        output_file = OUTPUT_DIR / f'{patient_id}_3d_landscape.html'
        fig.write_html(str(output_file))
        print(f"  ✓ {output_file.name}")
    
    def create_all_visualizations(self):
        """Create all 4 visualization types for all patients."""
        
        print("\n" + "="*80)
        print("STAGE 3: CREATING ALL SPATIAL VISUALIZATIONS")
        print("="*80)
        
        patients = self.data['patient_id'].unique()
        
        for patient_id in sorted(patients):
            print(f"\n{'='*80}")
            print(f"PATIENT: {patient_id}")
            print(f"{'='*80}")
            
            print(f"\nCreating spatial ecotype maps...")
            self.create_spatial_ecotype_map(patient_id)
            
            print(f"Creating confidence heatmaps...")
            self.create_confidence_heatmap(patient_id)
            
            print(f"Creating neighborhood analysis...")
            self.create_neighborhood_analysis(patient_id)
            
            print(f"Creating 3D tissue landscape...")
            self.create_3d_tissue_landscape(patient_id)
        
        print("\n" + "="*80)
        print("✅ ALL SPATIAL VISUALIZATIONS COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        
        # Count files
        png_count = len(list(OUTPUT_DIR.glob('*.png')))
        html_count = len(list(OUTPUT_DIR.glob('*.html')))
        
        print(f"Generated files:")
        print(f"  - PNG files: {png_count}")
        print(f"  - HTML files: {html_count}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    global PREDICTIONS_FILE, EMBEDDINGS_FILE, LABEL_MAPPING_FILE, METADATA_FILE, OUTPUT_DIR
    
    print("\n" + "="*80)
    print("SPATIAL LOCALIZATION VISUALIZATIONS - MODULAR PIPELINE")
    print("="*80)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate spatial visualization heatmaps and 3D landscapes'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml file (contains metadata and label mapping paths)'
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        help='Path to barcode metadata CSV file (optional if config provided)'
    )
    
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to fused embeddings NPY file (required)'
    )
    
    parser.add_argument(
        '--label-mapping',
        type=str,
        help='Path to label mapping JSON file (optional if config provided)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_dataset_config = config.get('input_dataset', {})
        
        # Get paths from config if not provided via command line
        # Resolve relative to CWD
        if not args.metadata:
            metadata_file = input_dataset_config.get('metadata_file', '')
            if metadata_file:
                metadata_path = Path(metadata_file)
                if not metadata_path.is_absolute():
                    metadata_path = Path.cwd() / metadata_file
                args.metadata = str(metadata_path)
        
        if not args.label_mapping:
            label_mapping_file = input_dataset_config.get('label_mapping_file', '')
            if label_mapping_file:
                label_mapping_path = Path(label_mapping_file)
                if not label_mapping_path.is_absolute():
                    label_mapping_path = Path.cwd() / label_mapping_file
                args.label_mapping = str(label_mapping_path)
    
    # Set global variables from arguments
    PREDICTIONS_FILE = args.predictions
    METADATA_FILE = args.metadata
    if not METADATA_FILE:
        print("ERROR: --metadata not provided and not found in config")
        sys.exit(1)
    
    EMBEDDINGS_FILE = args.embeddings  # Now required argument
    
    LABEL_MAPPING_FILE = args.label_mapping
    if not LABEL_MAPPING_FILE:
        print("ERROR: --label-mapping not provided and not found in config")
        sys.exit(1)
    
    OUTPUT_DIR = Path(args.output)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Configuration:")
    print(f"  Predictions:     {PREDICTIONS_FILE}")
    print(f"  Metadata:        {METADATA_FILE}")
    print(f"  Embeddings:      {EMBEDDINGS_FILE}")
    print(f"  Label Mapping:   {LABEL_MAPPING_FILE}")
    print(f"  Output Dir:      {OUTPUT_DIR}")
    
    viz = SpatialVisualizationPipeline()
    viz.load_data()
    viz.create_all_visualizations()
    
    print("\n✅ COMPLETE!")


if __name__ == '__main__':
    main()
