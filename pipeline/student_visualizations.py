#!/usr/bin/env python3
"""
Comprehensive Student Model Visualizations
============================================

Purpose:
    Generate publication-quality visualizations for student model results:
    - Preprocessing visualizations per patient
    - Spatial heatmaps with morphology-predicted ecotypes
    - Confidence maps and uncertainty regions
    - Neighborhood analysis with per-ecotype clustering
    - Teacher vs Student prediction comparisons
    - 3D tissue landscapes

Output Structure:
    student_visualizations/
    ├── preprocessing/
    │   ├── P1_embedding_distribution.png
    │   ├── P1_pca_variance.png
    │   └── ...
    ├── spatial/
    │   ├── P1_spatial_ecotype_map.png
    │   ├── P1_confidence_heatmap.png
    │   ├── P1_teacher_vs_student.png
    │   └── ...
    ├── neighborhood/
    │   ├── P1_neighborhood_analysis.png
    │   └── ...
    └── comparative/
        ├── all_patients_agreement_matrix.png
        ├── confidence_distribution.png
        └── ...

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
from typing import Dict, Tuple, Optional
import warnings
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===============================================================================
# COLOR SCHEME & CONSTANTS
# ===============================================================================

ECOTYPE_COLORS = {
    0: '#E74C3C',        # Fibrotic - Red
    1: '#3498DB',        # Immunosuppressive - Blue
    2: '#F39C12',        # Invasive Border - Yellow/Orange
    3: '#2ECC71',        # Metabolic - Green
    4: '#9B59B6'         # Normal Adjacent - Purple
}

ECOTYPE_NAMES = {
    0: 'Fibrotic',
    1: 'Immunosuppressive',
    2: 'Invasive_Border',
    3: 'Metabolic',
    4: 'Normal_Adjacent'
}

PATIENTS = ['P1', 'P2', 'P3', 'P4', 'P5']

# ===============================================================================
# STUDENT VISUALIZATION SUITE
# ===============================================================================

class StudentVisualizationSuite:
    """Generate comprehensive student model visualizations"""
    
    def __init__(self, output_dir: str, config: dict = None):
        """
        Initialize visualization suite.
        
        Args:
            output_dir: Base directory for saving visualizations
            config: Configuration dict with dataset info
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create visualization subdirectories
        self.vis_dirs = {
            'preprocessing': self.output_dir / 'preprocessing',
            'spatial': self.output_dir / 'spatial',
            'neighborhood': self.output_dir / 'neighborhood',
            'comparative': self.output_dir / 'comparative',
            '3d': self.output_dir / '3d_landscapes'
        }
        
        for dir_path in self.vis_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ StudentVisualizationSuite initialized")
        logger.info(f"  Output: {self.output_dir}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # HELPER: Normalize metadata column names
    # ═════════════════════════════════════════════════════════════════════════
    
    def _normalize_metadata_columns(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metadata column names to standard format.
        Converts 'x_coord'/'y_coord' to 'x'/'y' for compatibility.
        """
        metadata_df = metadata_df.copy()
        
        # Rename coordinate columns if needed
        if 'x_coord' in metadata_df.columns and 'x' not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={'x_coord': 'x'})
        if 'y_coord' in metadata_df.columns and 'y' not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={'y_coord': 'y'})
        
        # Ensure patient column exists
        if 'patient_id' in metadata_df.columns and 'patient' not in metadata_df.columns:
            metadata_df['patient'] = metadata_df['patient_id']
        
        return metadata_df
    
    # ═════════════════════════════════════════════════════════════════════════
    # 1. PREPROCESSING VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════════════
    
    def visualize_preprocessing(self, embeddings: np.ndarray, labels: np.ndarray,
                               metadata_df: pd.DataFrame = None,
                               pca_variance: float = None):
        """
        Create preprocessing visualization suite per patient.
        
        Args:
            embeddings: Preprocessed embeddings [N, D]
            labels: Class labels [N]
            metadata_df: Metadata with patient info
            pca_variance: PCA variance explained
        """
        logger.info("\n[PREPROCESSING VISUALIZATIONS]")
        logger.info("─" * 80)
        
        if metadata_df is None:
            logger.warning("  ⚠ No metadata provided, skipping per-patient visualization")
            return
        
        # Ensure metadata has patient info
        if 'patient' not in metadata_df.columns:
            logger.warning("  ⚠ No 'patient' column in metadata")
            return
        
        # Per-patient preprocessing
        for patient in PATIENTS:
            patient_mask = metadata_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() == 0:
                continue
            
            patient_emb = embeddings[patient_mask]
            patient_labels = labels[patient_mask]
            
            # 1. Embedding distribution
            self._plot_embedding_distribution(patient_emb, patient_labels, 
                                             patient, pca_variance)
            logger.info(f"  ✓ {patient}_embedding_distribution.png")
            
            # 2. PCA variance per patient
            self._plot_pca_variance(patient_emb, patient_labels, patient)
            logger.info(f"  ✓ {patient}_pca_variance.png")
            
            # 3. Class balance per patient
            self._plot_class_distribution(patient_labels, patient)
            logger.info(f"  ✓ {patient}_class_distribution.png")
    
    def _plot_embedding_distribution(self, embeddings: np.ndarray, labels: np.ndarray,
                                     patient: str, pca_variance: float = None):
        """Plot embedding distribution with class coloring"""
        
        from sklearn.decomposition import PCA
        
        # Project to 2D for visualization
        pca = PCA(n_components=2)
        embed_2d = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for class_id in np.unique(labels):
            mask = labels == class_id
            ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                      c=ECOTYPE_COLORS[class_id], label=ECOTYPE_NAMES[class_id],
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax.set_title(f'{patient}: Embedding Space (PCA-2D)', fontsize=13, fontweight='bold')
        ax.legend(title='Ecotype', fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['preprocessing'] / f'{patient}_embedding_distribution.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_variance(self, embeddings: np.ndarray, labels: np.ndarray,
                          patient: str, n_components: int = 50):
        """Plot PCA variance explained"""
        
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(n_components, embeddings.shape[1]))
        pca.fit(embeddings)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=6)
        axes[0].set_xlabel('Principal Component', fontsize=11)
        axes[0].set_ylabel('Variance Explained Ratio', fontsize=11)
        axes[0].set_title(f'{patient}: Scree Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(range(1, len(cum_var) + 1), cum_var, 'ro-', linewidth=2, markersize=6)
        axes[1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
        axes[1].set_xlabel('Number of Components', fontsize=11)
        axes[1].set_ylabel('Cumulative Variance Explained', fontsize=11)
        axes[1].set_title(f'{patient}: Cumulative Variance', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['preprocessing'] / f'{patient}_pca_variance.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, labels: np.ndarray, patient: str):
        """Plot class distribution"""
        
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar([ECOTYPE_NAMES[c] for c in unique], counts,
                      color=[ECOTYPE_COLORS[c] for c in unique],
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/len(labels)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{patient}: Class Distribution (N={len(labels)})',
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(counts) * 1.15])
        plt.xticks(rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['preprocessing'] / f'{patient}_class_distribution.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 2. SPATIAL VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════════════
    
    def visualize_spatial_predictions(self, predictions_df: pd.DataFrame,
                                     metadata_df: pd.DataFrame,
                                     label_encoder = None):
        """
        Create spatial heatmaps showing morphology-predicted ecotypes per patient.
        
        Args:
            predictions_df: DataFrame with barcode, prediction, confidence columns
            metadata_df: Metadata with spatial coordinates and true labels
            label_encoder: LabelEncoder for class names
        """
        logger.info("\n[SPATIAL VISUALIZATIONS]")
        logger.info("─" * 80)
        
        # Ensure spatial coordinates exist
        if not all(col in metadata_df.columns for col in ['x', 'y']):
            logger.warning("  ⚠ Missing spatial coordinates (x, y)")
            return
        
        # Per-patient spatial visualization
        for patient in PATIENTS:
            patient_mask = metadata_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() == 0:
                continue
            
            patient_meta = metadata_df[patient_mask].copy()
            patient_preds = predictions_df[patient_mask].copy()
            
            # Merge predictions with metadata
            patient_data = patient_meta.merge(
                patient_preds[['prediction', 'confidence', 'teacher_pred']],
                left_index=True, right_index=True, how='left'
            )
            
            # 1. Spatial ecotype map
            self._plot_spatial_ecotype_map(patient_data, patient, label_encoder)
            logger.info(f"  ✓ {patient}_spatial_ecotype_map.png")
            
            # 2. Confidence heatmap
            self._plot_confidence_heatmap(patient_data, patient)
            logger.info(f"  ✓ {patient}_confidence_heatmap.png")
            
            # 3. Teacher vs Student comparison
            if 'teacher_pred' in patient_preds.columns:
                self._plot_teacher_vs_student(patient_data, patient, label_encoder)
                logger.info(f"  ✓ {patient}_teacher_vs_student.png")
    
    def _plot_spatial_ecotype_map(self, patient_data: pd.DataFrame, patient: str,
                                  label_encoder = None):
        """Plot 3-panel: Ground truth | Student predictions | Accuracy map"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        x, y = patient_data['x'].values, patient_data['y'].values
        
        # Panel 1: Ground truth
        if 'label' in patient_data.columns:
            true_labels = patient_data['label'].values
            scatter1 = axes[0].scatter(x, y, c=true_labels, cmap='tab10',
                                      s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
            axes[0].set_title('Ground Truth Labels', fontsize=12, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'No true labels', ha='center', va='center',
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        
        # Panel 2: Student predictions
        pred_labels = patient_data['prediction'].values
        scatter2 = axes[1].scatter(x, y, c=pred_labels, cmap='tab10',
                                  s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
        axes[1].set_title('Student Predictions', fontsize=12, fontweight='bold')
        
        # Panel 3: Accuracy/Confidence
        if 'label' in patient_data.columns:
            accuracy = (pred_labels == patient_data['label'].values).astype(int)
            scatter3 = axes[2].scatter(x, y, c=accuracy, cmap='RdYlGn',
                                      s=30, alpha=0.8, edgecolors='black', linewidth=0.3,
                                      vmin=0, vmax=1)
            axes[2].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
            cbar3 = plt.colorbar(scatter3, ax=axes[2])
            cbar3.set_label('Correct')
        else:
            confidence = patient_data['confidence'].values
            scatter3 = axes[2].scatter(x, y, c=confidence, cmap='viridis',
                                      s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
            axes[2].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
            cbar3 = plt.colorbar(scatter3, ax=axes[2])
            cbar3.set_label('Confidence')
        
        for ax in axes:
            ax.set_xlabel('X Coordinate', fontsize=10)
            ax.set_ylabel('Y Coordinate', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
        
        fig.suptitle(f'{patient}: Spatial Ecotype Map (Morphology-Predicted)', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['spatial'] / f'{patient}_spatial_ecotype_map.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_heatmap(self, patient_data: pd.DataFrame, patient: str):
        """Plot confidence map and uncertainty regions"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x, y = patient_data['x'].values, patient_data['y'].values
        confidence = patient_data['confidence'].values
        
        # Panel 1: Confidence heatmap
        scatter1 = axes[0].scatter(x, y, c=confidence, cmap='RdYlGn',
                                  s=50, alpha=0.8, edgecolors='black', linewidth=0.3,
                                  vmin=0, vmax=1)
        axes[0].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X Coordinate', fontsize=10)
        axes[0].set_ylabel('Y Coordinate', fontsize=10)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.2)
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Confidence', fontsize=10)
        
        # Panel 2: Uncertainty regions (confidence < threshold)
        uncertainty_threshold = 0.5
        uncertainty = confidence < uncertainty_threshold
        
        # Create 3-level map: high conf | medium conf | low conf
        conf_levels = np.ones_like(confidence)
        conf_levels[uncertainty] = 0  # Low confidence = red
        conf_levels[(confidence >= uncertainty_threshold) & (confidence < 0.75)] = 1  # Medium = yellow
        conf_levels[confidence >= 0.75] = 2  # High = green
        
        colors_map = {0: '#E74C3C', 1: '#F39C12', 2: '#2ECC71'}
        conf_colors = [colors_map[int(c)] for c in conf_levels]
        
        scatter2 = axes[1].scatter(x, y, c=conf_colors, s=50, alpha=0.8,
                                  edgecolors='black', linewidth=0.3)
        axes[1].set_title('Uncertainty Regions', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X Coordinate', fontsize=10)
        axes[1].set_ylabel('Y Coordinate', fontsize=10)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', edgecolor='black', label=f'Low Conf (<{uncertainty_threshold})'),
            Patch(facecolor='#F39C12', edgecolor='black', label=f'Medium Conf ({uncertainty_threshold}-0.75)'),
            Patch(facecolor='#2ECC71', edgecolor='black', label='High Conf (>0.75)')
        ]
        axes[1].legend(handles=legend_elements, fontsize=10, loc='best')
        
        fig.suptitle(f'{patient}: Confidence Maps & Uncertainty Regions',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['spatial'] / f'{patient}_confidence_heatmap.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_teacher_vs_student(self, patient_data: pd.DataFrame, patient: str,
                                label_encoder = None):
        """Plot teacher vs student predictions comparison"""
        
        if 'teacher_pred' not in patient_data.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        x, y = patient_data['x'].values, patient_data['y'].values
        student_pred = patient_data['prediction'].values
        teacher_pred = patient_data['teacher_pred'].values
        
        # Panel 1: Teacher predictions
        scatter1 = axes[0, 0].scatter(x, y, c=teacher_pred, cmap='tab10',
                                     s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        axes[0, 0].set_title('Teacher Predictions', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X', fontsize=10)
        axes[0, 0].set_ylabel('Y', fontsize=10)
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.2)
        
        # Panel 2: Student predictions
        scatter2 = axes[0, 1].scatter(x, y, c=student_pred, cmap='tab10',
                                     s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        axes[0, 1].set_title('Student Predictions', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X', fontsize=10)
        axes[0, 1].set_ylabel('Y', fontsize=10)
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.2)
        
        # Panel 3: Agreement map (1=agree, 0=disagree)
        agreement = (student_pred == teacher_pred).astype(int)
        scatter3 = axes[1, 0].scatter(x, y, c=agreement, cmap='RdYlGn',
                                     s=40, alpha=0.8, edgecolors='black', linewidth=0.3,
                                     vmin=0, vmax=1)
        axes[1, 0].set_title(f'Agreement (Acc: {agreement.mean():.1%})',
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('X', fontsize=10)
        axes[1, 0].set_ylabel('Y', fontsize=10)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.2)
        cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
        cbar3.set_label('Agree')
        
        # Panel 4: Prediction difference magnitude
        diff = (student_pred.astype(float) - teacher_pred.astype(float))
        diff_magnitude = np.abs(diff)
        scatter4 = axes[1, 1].scatter(x, y, c=diff_magnitude, cmap='YlOrRd',
                                     s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        axes[1, 1].set_title('Prediction Difference Magnitude',
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('X', fontsize=10)
        axes[1, 1].set_ylabel('Y', fontsize=10)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.2)
        cbar4 = plt.colorbar(scatter4, ax=axes[1, 1])
        cbar4.set_label('Magnitude')
        
        fig.suptitle(f'{patient}: Teacher vs Student Predictions',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['spatial'] / f'{patient}_teacher_vs_student.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 3. NEIGHBORHOOD ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    
    def visualize_neighborhood_analysis(self, predictions_df: pd.DataFrame,
                                       metadata_df: pd.DataFrame):
        """
        Analyze neighborhoods: per-ecotype clustering and spatial patterns.
        
        Args:
            predictions_df: DataFrame with predictions
            metadata_df: Metadata with spatial coordinates
        """
        logger.info("\n[NEIGHBORHOOD ANALYSIS]")
        logger.info("─" * 80)
        
        if not all(col in metadata_df.columns for col in ['x', 'y']):
            logger.warning("  ⚠ Missing spatial coordinates")
            return
        
        # Per-patient neighborhood analysis
        for patient in PATIENTS:
            patient_mask = metadata_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() == 0:
                continue
            
            patient_meta = metadata_df[patient_mask].copy()
            patient_preds = predictions_df[patient_mask].copy()
            
            patient_data = patient_meta.merge(
                patient_preds[['prediction', 'confidence']],
                left_index=True, right_index=True, how='left'
            )
            
            self._plot_neighborhood_composition(patient_data, patient)
            logger.info(f"  ✓ {patient}_neighborhood_analysis.png")
    
    def _plot_neighborhood_composition(self, patient_data: pd.DataFrame, patient: str,
                                      n_neighbors: int = 10):
        """Analyze neighborhood composition for each ecotype"""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        x, y = patient_data['x'].values, patient_data['y'].values
        pred = patient_data['prediction'].values
        
        coords = np.column_stack([x, y])
        
        # Build KNN tree
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Per-ecotype neighborhood composition
        for class_id in range(5):
            ax = axes[class_id]
            
            class_mask = pred == class_id
            if class_mask.sum() == 0:
                ax.text(0.5, 0.5, f'{ECOTYPE_NAMES[class_id]}\n(N=0)',
                       ha='center', va='center', fontsize=11, transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Get neighborhood composition for class cells
            class_indices = np.where(class_mask)[0]
            neighbor_composition = []
            
            for idx in class_indices:
                # Get neighbors (excluding self)
                neighbor_idx = indices[idx][1:]
                neighbor_classes = pred[neighbor_idx]
                comp, _ = np.histogram(neighbor_classes, bins=5, range=(0, 5))
                neighbor_composition.append(comp)
            
            neighbor_composition = np.array(neighbor_composition).mean(axis=0)
            neighbor_composition = neighbor_composition / neighbor_composition.sum()
            
            # Plot neighborhood composition
            colors = [ECOTYPE_COLORS[i] for i in range(5)]
            bars = ax.bar([ECOTYPE_NAMES[i][:4] for i in range(5)],
                         neighbor_composition, color=colors, edgecolor='black',
                         linewidth=1.5, alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, neighbor_composition):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1%}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Composition', fontsize=10)
            ax.set_title(f'{ECOTYPE_NAMES[class_id]}\nNeighbors (N={class_mask.sum()})',
                        fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 6: Neighborhood diversity index
        ax = axes[5]
        diversity_per_class = []
        
        for class_id in range(5):
            class_mask = pred == class_id
            if class_mask.sum() == 0:
                diversity_per_class.append(0)
                continue
            
            class_indices = np.where(class_mask)[0]
            diversity_scores = []
            
            for idx in class_indices:
                neighbor_idx = indices[idx][1:]
                neighbor_classes = pred[neighbor_idx]
                # Shannon entropy as diversity measure
                unique, counts = np.unique(neighbor_classes, return_counts=True)
                probs = counts / len(neighbor_classes)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                diversity_scores.append(entropy)
            
            diversity_per_class.append(np.mean(diversity_scores))
        
        bars = ax.bar([ECOTYPE_NAMES[i][:4] for i in range(5)],
                     diversity_per_class, 
                     color=[ECOTYPE_COLORS[i] for i in range(5)],
                     edgecolor='black', linewidth=1.5, alpha=0.7)
        
        for bar, val in zip(bars, diversity_per_class):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Shannon Entropy', fontsize=10)
        ax.set_title('Neighborhood Diversity\n(Higher = More Mixed)', fontsize=11, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'{patient}: Neighborhood Analysis (K={n_neighbors})',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['neighborhood'] / f'{patient}_neighborhood_analysis.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 4. COMPARATIVE ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    
    def visualize_comparative_analysis(self, predictions_df: pd.DataFrame,
                                      metadata_df: pd.DataFrame,
                                      label_encoder = None):
        """
        Create comparative visualizations across all patients.
        
        Args:
            predictions_df: DataFrame with predictions
            metadata_df: Metadata
            label_encoder: LabelEncoder for class names
        """
        logger.info("\n[COMPARATIVE VISUALIZATIONS]")
        logger.info("─" * 80)
        
        # 1. Agreement matrix across patients
        self._plot_agreement_matrix(predictions_df, metadata_df)
        logger.info(f"  ✓ teacher_student_agreement_matrix.png")
        
        # 2. Confidence distribution
        self._plot_confidence_distribution(predictions_df, metadata_df)
        logger.info(f"  ✓ confidence_distribution.png")
        
        # 3. Prediction distribution per class
        self._plot_prediction_distribution(predictions_df, label_encoder)
        logger.info(f"  ✓ prediction_distribution_per_class.png")
    
    def _plot_agreement_matrix(self, predictions_df: pd.DataFrame, metadata_df: pd.DataFrame):
        """Plot agreement between teacher and student per patient"""
        
        if 'teacher_pred' not in predictions_df.columns:
            return
        
        # CRITICAL FIX: Use proper barcode-based merging instead of position-based
        # Merge predictions with metadata on barcode to get patient info
        if 'barcode' in predictions_df.columns and 'barcode' in metadata_df.columns:
            merged_df = predictions_df.merge(
                metadata_df[['barcode', 'patient']],
                on='barcode',
                how='left'
            )
            if 'patient' not in merged_df.columns:
                logger.warning("  ⚠ Could not merge predictions with patient info by barcode")
                return
        else:
            logger.warning("  ⚠ Missing barcode columns for proper merging")
            return
        
        agreement_by_patient = []
        patient_names = []
        
        for patient in PATIENTS:
            patient_mask = merged_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() == 0:
                continue
            
            patient_preds = merged_df[patient_mask]
            
            if len(patient_preds) > 0:
                agreement = (patient_preds['prediction'] == patient_preds['teacher_pred']).mean()
                agreement_by_patient.append(agreement)
                patient_names.append(patient)
                logger.info(f"  {patient}: {len(patient_preds)} samples, {agreement:.1%} agreement")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ECC71' if a > 0.8 else '#F39C12' if a > 0.7 else '#E74C3C'
                 for a in agreement_by_patient]
        
        bars = ax.bar(patient_names, agreement_by_patient, color=colors,
                     edgecolor='black', linewidth=2, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, agreement_by_patient):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='80% threshold')
        ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='70% threshold')
        
        ax.set_ylabel('Agreement Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Patient', fontsize=12, fontweight='bold')
        ax.set_title('Teacher-Student Prediction Agreement by Patient',
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['comparative'] / 'teacher_student_agreement_matrix.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, predictions_df: pd.DataFrame,
                                     metadata_df: pd.DataFrame):
        """Plot confidence distribution across patients"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confidence histogram
        confidence = predictions_df['confidence'].values
        axes[0].hist(confidence, bins=30, color='#3498DB', edgecolor='black',
                    alpha=0.7, linewidth=1.5)
        axes[0].axvline(x=confidence.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {confidence.mean():.2f}')
        axes[0].axvline(x=np.median(confidence), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(confidence):.2f}')
        axes[0].set_xlabel('Confidence', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[0].set_title('Overall Confidence Distribution',
                         fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Per-patient boxplot
        confidence_by_patient = []
        patient_labels = []
        
        for patient in PATIENTS:
            patient_mask = metadata_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() == 0:
                continue
            conf = predictions_df[patient_mask]['confidence'].values
            confidence_by_patient.append(conf)
            patient_labels.append(patient)
        
        bp = axes[1].boxplot(confidence_by_patient, labels=patient_labels,
                            patch_artist=True, notch=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Patient', fontsize=11, fontweight='bold')
        axes[1].set_title('Confidence Distribution by Patient',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Student Model Confidence Analysis', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['comparative'] / 'confidence_distribution.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, predictions_df: pd.DataFrame,
                                     label_encoder = None):
        """Plot prediction distribution per class"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pred_counts = np.bincount(predictions_df['prediction'].astype(int), minlength=5)
        pred_pcts = pred_counts / len(predictions_df) * 100
        
        class_names = [ECOTYPE_NAMES[i][:12] for i in range(5)]
        colors = [ECOTYPE_COLORS[i] for i in range(5)]
        
        bars = ax.bar(class_names, pred_pcts, color=colors, edgecolor='black',
                     linewidth=2, alpha=0.7)
        
        # Add value labels
        for bar, count, pct in zip(bars, pred_counts, pred_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Student Model Prediction Distribution',
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(pred_pcts) * 1.15])
        plt.xticks(rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.vis_dirs['comparative'] / 'prediction_distribution_per_class.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═════════════════════════════════════════════════════════════════════════
    
    def generate_summary_report(self, predictions_df: pd.DataFrame,
                               metadata_df: pd.DataFrame):
        """Generate summary visualization report"""
        
        logger.info("\n[GENERATING SUMMARY REPORT]")
        logger.info("─" * 80)
        
        report = {
            'total_spots': len(predictions_df),
            'avg_confidence': float(predictions_df['confidence'].mean()),
            'confidence_std': float(predictions_df['confidence'].std()),
            'patients_analyzed': []
        }
        
        # Per-patient stats
        for patient in PATIENTS:
            patient_mask = metadata_df['patient'].str.contains(patient, case=False, na=False)
            if patient_mask.sum() > 0:
                patient_preds = predictions_df[patient_mask]
                
                patient_stats = {
                    'patient': patient,
                    'n_spots': len(patient_preds),
                    'avg_confidence': float(patient_preds['confidence'].mean()),
                    'low_confidence_pct': float((patient_preds['confidence'] < 0.5).mean() * 100)
                }
                
                # Add teacher agreement if available
                if 'teacher_pred' in patient_preds.columns:
                    patient_stats['teacher_agreement'] = float(
                        (patient_preds['prediction'] == patient_preds['teacher_pred']).mean()
                    )
                
                report['patients_analyzed'].append(patient_stats)
        
        # Save report
        report_path = self.vis_dirs['comparative'] / 'visualization_summary.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"  ✓ visualization_summary.json")
        
        return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_student_visualizations(
    predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    config: dict = None,
    label_encoder = None,
    pca_variance: float = None
):
    """
    Generate complete student visualization suite.
    
    Args:
        predictions_df: Student predictions
        metadata_df: Spatial metadata
        embeddings: Preprocessed embeddings
        labels: True labels
        output_dir: Output directory
        config: Config dict
        label_encoder: LabelEncoder for classes
        pca_variance: PCA variance explained
    """
    
    # Create visualization suite
    viz_suite = StudentVisualizationSuite(output_dir, config)
    
    # Normalize metadata columns (convert x_coord->x, y_coord->y, etc.)
    if metadata_df is not None:
        metadata_df = viz_suite._normalize_metadata_columns(metadata_df)
    
    # 1. Preprocessing
    viz_suite.visualize_preprocessing(embeddings, labels, metadata_df, pca_variance)
    
    # 2. Spatial
    viz_suite.visualize_spatial_predictions(predictions_df, metadata_df, label_encoder)
    
    # 3. Neighborhood
    viz_suite.visualize_neighborhood_analysis(predictions_df, metadata_df)
    
    # 4. Comparative
    viz_suite.visualize_comparative_analysis(predictions_df, metadata_df, label_encoder)
    
    # 5. Summary
    report = viz_suite.generate_summary_report(predictions_df, metadata_df)
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL VISUALIZATIONS COMPLETE")
    logger.info(f"   Output directory: {output_dir}")
    logger.info("="*80)
    
    return viz_suite, report


if __name__ == "__main__":
    logger.info("Student Visualization Suite - Ready for integration")
