#!/usr/bin/env python3
"""
Post-Training Visualizations
=============================

Purpose:
  Generate comprehensive training and spatial visualization suite after model training

Features:
  - Training curves (loss per fold with early stopping indicator)
  - Global confusion matrix (all 23,342 spots)
  - Moran's I spatial analysis per patient
  - 3D tissue landscape visualization
  - Neighborhood analysis with per-ecotype clustering
  - Combined spatial ecotype mapping (GT vs Predictions)

Color Scheme:
  - Fibrotic: Red (#E74C3C)
  - Metabolic: Bright Green (#2ECC71)
  - Immunosuppressive: Blue (#3498DB)
  - Normal Adjacent: Purple (#9B59B6)
  - Invasive Border: Yellow (#F39C12)

Usage:
  viz = PostTrainingVisualizer(
      results_dir='results/training',
      output_dir='results/visualizations',
      config=config
  )
  
  # Generate all visualizations
  viz.generate_all_visualizations(
      predictions_df=predictions_df,
      fold_histories=fold_histories
  )

Author: Visualization Pipeline
Date: March 24, 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════════════════

ECOTYPE_COLORS = {
    0: '#E74C3C',        # Fibrotic - Red
    1: '#3498DB',        # Immunosuppressive - Blue
    2: '#F39C12',        # Invasive Border - Yellow/Orange
    3: '#2ECC71',        # Metabolic - Bright Green
    4: '#9B59B6'         # Normal Adjacent - Purple
}

ECOTYPE_NAMES = {
    0: 'Fibrotic',
    1: 'Immunosuppressive',
    2: 'Invasive_Border',
    3: 'Metabolic',
    4: 'Normal_Adjacent'
}

ECOTYPE_SHORT = {
    0: 'Fib',
    1: 'Imm',
    2: 'Inv',
    3: 'Met',
    4: 'Nor'
}


class PostTrainingVisualizer:
    """
    Generate and save all post-training visualizations.
    """
    
    def __init__(self, results_dir: str, output_dir: str, config: dict = None):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory with training results (metrics)
            output_dir: Directory to save visualizations
            config: Configuration dict with dataset info
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        
        # Create subdirectories
        self.visualizations = {
            'training': self.output_dir / 'training_curves',
            'confusion': self.output_dir / 'confusion_matrices',
            'spatial': self.output_dir / 'spatial_analysis',
            'landscape': self.output_dir / '3d_landscapes',
            'neighborhood': self.output_dir / 'neighborhood_analysis'
        }
        
        for vis_dir in self.visualizations.values():
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ PostTrainingVisualizer initialized")
        print(f"  Output directory: {self.output_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING CURVES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_training_curves(self, fold_histories: Dict, figsize: Tuple = (16, 10)):
        """
        Plot training curves for all folds with early stopping indicators.
        
        Args:
            fold_histories: Dict mapping fold_id → {'epoch': [...], 'train_loss': [...], 'val_loss': [...], ...}
            figsize: Figure size
        """
        print("\n📈 Plotting training curves...")
        
        num_folds = len(fold_histories)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for fold_id in range(num_folds):
            if fold_id not in fold_histories:
                continue
            
            history = fold_histories[fold_id]
            ax = axes[fold_id]
            
            epochs = history.get('epoch', [])
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            best_epoch = history.get('best_epoch', 0)
            early_stopped = history.get('early_stopped', False)
            
            if not epochs:
                continue
            
            # Plot losses
            ax.plot(epochs, train_loss, 'o-', label='Training Loss', 
                   linewidth=2, markersize=4, color='#3498DB')
            ax.plot(epochs, val_loss, 's-', label='Validation Loss', 
                   linewidth=2, markersize=4, color='#E74C3C')
            
            # Mark best epoch
            if best_epoch > 0 and best_epoch <= len(epochs):
                best_val_loss = val_loss[best_epoch - 1]
                ax.plot(best_epoch, best_val_loss, '*', 
                       markersize=20, color='#F39C12', 
                       label=f'Best (Epoch {best_epoch})', zorder=5)
            
            # Mark early stopping
            if early_stopped:
                ax.axvline(x=best_epoch, color='#E74C3C', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Early Stop')
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax.set_title(f'Fold {fold_id} Training History', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if num_folds < 6:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        output_file = self.visualizations['training'] / 'training_curves.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFUSION MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             figsize: Tuple = (12, 10)):
        """
        Plot global confusion matrix across all 23,342 spots.
        
        Args:
            y_true: Ground truth labels (shape: N,)
            y_pred: Predicted labels (shape: N,)
            figsize: Figure size
        """
        print("\n🔲 Plotting confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=[ECOTYPE_SHORT[i] for i in range(5)])
        disp.plot(ax=ax, cmap='Blues', values_format='d', im_kws={'cmap': 'Blues'})
        
        # Format
        ax.set_title('Global Confusion Matrix\n(All 23,342 Spots × 5 Classes)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth Label', fontsize=12, fontweight='bold')
        
        # Add accuracy annotation
        accuracy = np.trace(cm) / cm.sum()
        textstr = f'Overall Accuracy: {accuracy:.4f}'
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        output_file = self.visualizations['confusion'] / 'global_confusion_matrix.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
        
        # Save normalized confusion matrix
        self._plot_normalized_confusion_matrix(cm, figsize)
    
    def _plot_normalized_confusion_matrix(self, cm: np.ndarray, figsize: Tuple):
        """Plot normalized confusion matrix (by true label)."""
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                      display_labels=[ECOTYPE_SHORT[i] for i in range(5)])
        disp.plot(ax=ax, cmap='Greens', values_format='.3f', im_kws={'cmap': 'Greens'})
        
        ax.set_title('Normalized Confusion Matrix\n(Recall per Class)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.visualizations['confusion'] / 'normalized_confusion_matrix.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MORAN'S I SPATIAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_morans_i(self, spatial_coords: np.ndarray, 
                          labels: np.ndarray, k_neighbors: int = 6) -> float:
        """
        Calculate Moran's I statistic for spatial autocorrelation.
        
        Args:
            spatial_coords: Array of shape (N, 2) with x, y coordinates
            labels: Array of shape (N,) with class labels (0-4)
            k_neighbors: Number of nearest neighbors to consider
        
        Returns:
            Moran's I value (range: -1 to 1, >0.2 indicates clustering)
        """
        if len(spatial_coords) < k_neighbors + 1:
            return 0.0
        
        # Find k-nearest neighbors
        distances = cdist(spatial_coords, spatial_coords)
        
        # Create weight matrix (1 if neighbor, 0 otherwise)
        W = np.zeros_like(distances)
        for i in range(len(spatial_coords)):
            nearest_indices = np.argsort(distances[i])[1:k_neighbors+1]
            W[i, nearest_indices] = 1
        
        # Normalize weights
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
        
        # Convert labels to numeric
        y = labels.astype(float)
        y_mean = y.mean()
        
        # Calculate Moran's I
        numerator = 0.0
        for i in range(len(y)):
            for j in range(len(y)):
                numerator += W[i, j] * (y[i] - y_mean) * (y[j] - y_mean)
        
        denominator = np.sum((y - y_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        n = len(y)
        morans_i = (n / W.sum()) * (numerator / denominator)
        
        return morans_i
    
    def plot_spatial_morans_i(self, spatial_coords: np.ndarray, 
                             predictions: np.ndarray, ground_truth: np.ndarray,
                             patient_id: str, figsize: Tuple = (16, 6)):
        """
        Plot Moran's I analysis for a patient (Ground Truth vs Predictions).
        
        Args:
            spatial_coords: Array shape (N, 2) with x, y coordinates
            predictions: Predicted labels (0-4)
            ground_truth: Ground truth labels (0-4)
            patient_id: Patient identifier (e.g., 'P1')
            figsize: Figure size
        """
        print(f"\n📊 Plotting Moran's I analysis for {patient_id}...")
        
        # Calculate Moran's I
        morans_gt = self.calculate_morans_i(spatial_coords, ground_truth)
        morans_pred = self.calculate_morans_i(spatial_coords, predictions)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Ground Truth
        ax = axes[0]
        scatter = ax.scatter(
            spatial_coords[:, 0], spatial_coords[:, 1],
            c=ground_truth, cmap='tab10', s=30, alpha=0.7, edgecolors='black', linewidth=0.3
        )
        ax.set_xlabel('Array Column', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Row', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient_id} - Ground Truth\nMoran\'s I: {morans_gt:.4f}',
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Label', fontsize=10, fontweight='bold')
        
        # Predictions
        ax = axes[1]
        scatter = ax.scatter(
            spatial_coords[:, 0], spatial_coords[:, 1],
            c=predictions, cmap='tab10', s=30, alpha=0.7, edgecolors='black', linewidth=0.3
        )
        ax.set_xlabel('Array Column', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Row', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient_id} - Predictions\nMoran\'s I: {morans_pred:.4f}',
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Label', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.visualizations['spatial'] / f'{patient_id}_morans_i_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
        
        return morans_gt, morans_pred
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINED SPATIAL ECOTYPE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_spatial_ecotype_map(self, spatial_coords: np.ndarray,
                                predictions: np.ndarray, ground_truth: np.ndarray,
                                patient_id: str, figsize: Tuple = (22, 8)):
        """
        Plot 3-panel spatial ecotype map: Ground Truth, Predictions, Accuracy.
        
        Args:
            spatial_coords: Array shape (N, 2) with x, y coordinates
            predictions: Predicted labels (0-4)
            ground_truth: Ground truth labels (0-4)
            patient_id: Patient identifier
            figsize: Figure size
        """
        print(f"\n🗺️  Plotting spatial ecotype map for {patient_id}...")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel 1: Ground Truth
        ax = axes[0]
        for ecotype_id in range(5):
            mask = ground_truth == ecotype_id
            ax.scatter(
                spatial_coords[mask, 0], spatial_coords[mask, 1],
                c=ECOTYPE_COLORS[ecotype_id],
                label=ECOTYPE_NAMES[ecotype_id],
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5
            )
        ax.set_xlabel('Array Column', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Row', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient_id} - Ground Truth', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Predictions
        ax = axes[1]
        for ecotype_id in range(5):
            mask = predictions == ecotype_id
            ax.scatter(
                spatial_coords[mask, 0], spatial_coords[mask, 1],
                c=ECOTYPE_COLORS[ecotype_id],
                label=ECOTYPE_NAMES[ecotype_id],
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5
            )
        accuracy = (ground_truth == predictions).mean() * 100
        ax.set_xlabel('Array Column', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Row', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient_id} - Predictions\n(Accuracy: {accuracy:.2f}%)',
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Accuracy Map
        ax = axes[2]
        correct = ground_truth == predictions
        ax.scatter(
            spatial_coords[correct, 0], spatial_coords[correct, 1],
            c='#2ECC71', label=f'Correct (n={correct.sum()})',
            s=60, alpha=0.6, edgecolors='darkgreen', linewidth=0.5, marker='o'
        )
        ax.scatter(
            spatial_coords[~correct, 0], spatial_coords[~correct, 1],
            c='#E74C3C', label=f'Errors (n={(~correct).sum()})',
            s=100, alpha=0.8, edgecolors='darkred', linewidth=1, marker='X'
        )
        ax.set_xlabel('Array Column', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Row', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient_id} - Accuracy Map\n({accuracy:.2f}%)',
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.visualizations['spatial'] / f'{patient_id}_spatial_ecotype_map.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
        
        return accuracy
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEIGHBORHOOD ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_neighborhood_analysis(self, spatial_coords: np.ndarray,
                                  predictions: np.ndarray,
                                  patient_id: str, k_neighbors: int = 6,
                                  figsize: Tuple = (20, 12)):
        """
        Plot neighborhood composition for each ecotype.
        
        Shows: For each ecotype, what are the surrounding neighbors?
        
        Args:
            spatial_coords: Array shape (N, 2)
            predictions: Predicted labels (0-4)
            patient_id: Patient identifier
            k_neighbors: Number of neighbors to consider
            figsize: Figure size
        """
        print(f"\n👥 Plotting neighborhood analysis for {patient_id}...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Find neighbors
        neigh = NearestNeighbors(n_neighbors=k_neighbors+1)
        neigh.fit(spatial_coords)
        distances, indices = neigh.kneighbors(spatial_coords)
        
        # For each ecotype, count neighbor compositions
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for ecotype_id in range(5):
            ax = axes[ecotype_id]
            
            # Get spots of this ecotype
            spots = np.where(predictions == ecotype_id)[0]
            
            if len(spots) == 0:
                ax.text(0.5, 0.5, f'No {ECOTYPE_NAMES[ecotype_id]} spots',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{ECOTYPE_NAMES[ecotype_id]}', fontsize=12, fontweight='bold')
                continue
            
            # Count neighbor ecotypes
            neighbor_counts = np.zeros((5, k_neighbors))
            
            for spot_idx, spot in enumerate(spots):
                neighbor_indices = indices[spot, 1:]  # Exclude self
                neighbor_labels = predictions[neighbor_indices]
                
                for ni, label in enumerate(neighbor_labels):
                    neighbor_counts[label, ni] += 1
            
            # Calculate average composition
            total_neighbors = neighbor_counts.sum()
            if total_neighbors > 0:
                neighbor_pcts = 100 * neighbor_counts.sum(axis=1) / total_neighbors
            else:
                neighbor_pcts = np.zeros(5)
            
            # Plot bar chart
            colors = [ECOTYPE_COLORS[i] for i in range(5)]
            bars = ax.bar(range(5), neighbor_pcts, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Neighbor Percentage (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{ECOTYPE_NAMES[ecotype_id]} Neighborhood\n(n={len(spots)} spots)',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(5))
            ax.set_xticklabels([ECOTYPE_SHORT[i] for i in range(5)], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(neighbor_pcts) * 1.15)
        
        # Hide last subplot
        fig.delaxes(axes[5])
        
        # Add overall title
        fig.suptitle(f'{patient_id} - Neighborhood Composition\n(k={k_neighbors} nearest neighbors)',
                    fontsize=14, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        output_file = self.visualizations['neighborhood'] / f'{patient_id}_neighborhood_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3D LANDSCAPE VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_3d_landscape(self, spatial_coords: np.ndarray,
                         predictions: np.ndarray, z_values: np.ndarray,
                         patient_id: str, z_label: str = 'Cell Composition'):
        """
        Create interactive 3D scatter plot.
        
        Args:
            spatial_coords: Array shape (N, 2) with x, y
            predictions: Predicted labels (0-4)
            z_values: Array shape (N,) with z-axis values
            patient_id: Patient identifier
            z_label: Label for Z axis
        """
        if not PLOTLY_AVAILABLE:
            print(f"⚠ Plotly not available, skipping 3D landscape for {patient_id}")
            return
        
        print(f"\n🌄 Plotting 3D landscape for {patient_id}...")
        
        # Create hover text
        hover_text = [
            f"Ecotype: {ECOTYPE_NAMES[pred]}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
            for pred, x, y, z in zip(predictions, spatial_coords[:, 0], spatial_coords[:, 1], z_values)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter for each ecotype
        for ecotype_id in range(5):
            mask = predictions == ecotype_id
            fig.add_trace(go.Scatter3d(
                x=spatial_coords[mask, 0],
                y=spatial_coords[mask, 1],
                z=z_values[mask],
                mode='markers',
                name=ECOTYPE_NAMES[ecotype_id],
                marker=dict(
                    size=4,
                    color=ECOTYPE_COLORS[ecotype_id],
                    opacity=0.7,
                    line=dict(color='black', width=0.5)
                ),
                text=[hover_text[i] for i in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{patient_id} - 3D Tissue Landscape ({z_label})',
            scene=dict(
                xaxis_title='Array Column',
                yaxis_title='Array Row',
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=800,
            width=1000,
            showlegend=True
        )
        
        # Save
        output_file = self.visualizations['landscape'] / f'{patient_id}_3d_landscape.html'
        fig.write_html(str(output_file))
        print(f"✅ Saved: {output_file.name}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_visualization_metrics(self, metrics_dict: Dict, filename: str = 'visualization_metrics.json'):
        """
        Save all calculated metrics to JSON.
        
        Args:
            metrics_dict: Dictionary with all metrics
            filename: Output filename
        """
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"✓ Saved visualization metrics: {output_file.name}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ORCHESTRATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_all_visualizations(self, predictions_df: pd.DataFrame,
                                   fold_histories: Dict = None,
                                   y_true: np.ndarray = None,
                                   y_pred: np.ndarray = None):
        """
        Generate all visualizations from predictions.
        
        Args:
            predictions_df: DataFrame with columns:
                - patient_id, array_col, array_row, predicted_label, ground_truth_label, ...
            fold_histories: Dict mapping fold_id → history dict (optional)
            y_true: Global ground truth labels (optional)
            y_pred: Global predictions (optional)
        """
        print("\n" + "="*80)
        print("GENERATING ALL POST-TRAINING VISUALIZATIONS")
        print("="*80)
        
        metrics_dict = {}
        
        # 1. Training curves (if histories provided)
        if fold_histories:
            self.plot_training_curves(fold_histories)
        
        # 2. Global confusion matrix
        if y_true is None or y_pred is None:
            y_true = predictions_df['ground_truth_label'].values
            y_pred = predictions_df['predicted_label'].values
        
        self.plot_confusion_matrix(y_true, y_pred)
        
        # 3. Per-patient visualizations
        patients = predictions_df['patient_id'].unique()
        morans_results = {}
        accuracy_results = {}
        
        for patient_id in sorted(patients):
            patient_data = predictions_df[predictions_df['patient_id'] == patient_id]
            
            # Extract data
            spatial_coords = patient_data[['array_col', 'array_row']].values
            predictions = patient_data['predicted_label'].values
            ground_truth = patient_data['ground_truth_label'].values
            
            if len(spatial_coords) < 10:
                print(f"⚠ Skipping {patient_id} (too few spots: {len(spatial_coords)})")
                continue
            
            # Plot spatial ecotype map
            acc = self.plot_spatial_ecotype_map(spatial_coords, predictions, 
                                               ground_truth, patient_id)
            accuracy_results[patient_id] = float(acc)
            
            # Plot Moran's I analysis
            morans_gt, morans_pred = self.plot_spatial_morans_i(
                spatial_coords, predictions, ground_truth, patient_id
            )
            morans_results[patient_id] = {
                'ground_truth': float(morans_gt),
                'predictions': float(morans_pred)
            }
            
            # Plot neighborhood analysis
            self.plot_neighborhood_analysis(spatial_coords, predictions, patient_id)
            
            # Plot 3D landscape (if cell composition data available)
            if 'cell_composition' in patient_data.columns:
                z_values = patient_data['cell_composition'].values
                self.plot_3d_landscape(spatial_coords, predictions, z_values, patient_id)
        
        # Save metrics
        metrics_dict['morans_i'] = morans_results
        metrics_dict['accuracy_per_patient'] = accuracy_results
        self.save_visualization_metrics(metrics_dict)
        
        print("\n" + "="*80)
        print("✅ ALL VISUALIZATIONS COMPLETE")
        print("="*80)
