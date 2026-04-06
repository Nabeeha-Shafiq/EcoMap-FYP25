#!/usr/bin/env python3
"""
Unified Student Model Training with Knowledge Distillation + Visualizations

Purpose:
    Complete student training pipeline integrated with teacher:
    1. Load preprocessed embeddings (UNI from teacher preprocessing)
    2. Load ensemble teacher model  
    3. Train student with 5-fold CV + distillation loss
    4. Generate comprehensive visualizations (matching teacher quality)
    5. Save metrics and post-training analysis

Usage:
    python pipeline/train_student_model_unified.py --config config/modular_Student_GEO.yaml

Expected Output Structure:
    ./GEO Ablation Study/STUDENT_DISTILLATION_0.6_PCA/
    ├── preprocessing/
    │   └── student_pca_model.pkl
    ├── training/
    │   ├── models/
    │   │   ├── fold_0_best_student_model.pth
    │   │   └── ... (fold 1-4)
    │   ├── metrics/
    │   │   ├── training_results.json
    │   │   ├── fold_results.csv
    │   │   ├── predictions_all_spots.csv
    │   │   └── label_encoder.pkl
    │   └── visualizations/
    │       ├── training_curves.png
    │       ├── confusion_matrix.png
    │       ├── per_class_accuracy.png
    │       └── fold_comparison.png
    └── post-training/
        ├── metrics/
        │   └── post_training_metrics.json
        └── visualizations/
            ├── P1_spatial_ecotype_map.png
            ├── P1_confidence_heatmap.png
            ├── P1_neighborhood_analysis.png
            └── ... (for each patient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pandas as pd
import json
import yaml
import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from datetime import datetime
import logging
import pickle
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.lines import Line2D

# Import student visualization suite
try:
    from student_visualizations import generate_all_student_visualizations
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠ Warning: student_visualizations module not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# TEACHER MODEL ARCHITECTURE
# ============================================================================

class EcotypeClassifier(nn.Module):
    """Teacher model architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# STUDENT MODEL ARCHITECTURE
# ============================================================================

class StudentMLPClassifier(nn.Module):
    """Simple MLP for student (morphology-only)"""
    
    def __init__(self, input_dim, hidden_dims, n_classes, dropout_rate=0.3):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [n_classes]
        layers = []
        
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

class KnowledgeDistillationLoss(nn.Module):
    """Three-component distillation loss"""
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, student_features, 
                teacher_features, labels):
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_log_soft, teacher_soft)
        
        # Feature matching (skip if dimensions mismatch)
        if student_features.shape == teacher_features.shape:
            feature_loss = torch.nn.functional.mse_loss(student_features, teacher_features.detach())
        else:
            feature_loss = torch.tensor(0.0, device=student_features.device, dtype=student_features.dtype)
        
        total_loss = self.alpha * hard_loss + self.beta * soft_loss + self.gamma * feature_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'hard': hard_loss.item(),
            'soft': soft_loss.item(),
            'feature': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else 0.0
        }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_data(config: Dict):
    """Load and preprocess student data using teacher's PCA models"""
    
    print("\n[1] Loading data...")
    
    # Load UNI embeddings (raw) - PRESERVE BARCODE ORDER
    embeddings_path = Path(config['input_dataset']['image_encoder_embeddings'])
    embeddings_df = pd.read_csv(embeddings_path)
    barcode_indices = embeddings_df.iloc[:, 0].values  # First column is barcode/index
    embeddings_array = embeddings_df.iloc[:, 1:].values.astype(np.float32)
    
    input_dim = embeddings_array.shape[1]
    
    # Determine which modality is being used
    embedding_file = embeddings_path.name
    if 'uni' in embedding_file.lower():
        modality = "UNI (Morphology)"
    elif 'scvi' in embedding_file.lower():
        modality = "scVI (Gene Expression)"
    elif 'rctd' in embedding_file.lower() or 'cell' in embedding_file.lower():
        modality = "RCTD (Cell Composition)"
    else:
        modality = "Unknown"
    
    print(f"✓ Student embedding modality: {modality}")
    print(f"✓ Loaded embeddings from: {embedding_file}")
    print(f"✓ Shape: {embeddings_array.shape} ({input_dim}D)")
    
    # Load labels
    labels_path = Path(config['input_dataset']['labels_file'])
    labels_df = pd.read_csv(labels_path)
    labels = labels_df.iloc[:, 1].values
    
    print(f"✓ Loaded labels: {len(np.unique(labels))} classes, {len(labels)} samples")
    
    # CRITICAL: Load teacher's PCA model (ensure consistency with teacher)
    teacher_preproc_dir = Path(config['distillation']['teacher_preprocessed_dir'])
    teacher_pca_path = teacher_preproc_dir / "pca_models" / "pca_image_encoder.pkl"
    
    pca_var = config['embeddings']['image_encoder']['pca_variance']
    
    if teacher_pca_path.exists():
        # Check if teacher's PCA model matches current input dimensions
        with open(teacher_pca_path, 'rb') as f:
            pca = pickle.load(f)
        
        # Check PCA input dimension compatibility
        pca_input_dim = pca.n_features_in_
        
        if pca_input_dim == input_dim:
            # Dimensions match: apply teacher's PCA model for consistency
            embeddings_array = pca.transform(embeddings_array)
            print(f"✓ Applied teacher's PCA model: {input_dim}D → {embeddings_array.shape[1]}D")
            print(f"  (Dimensions matched, using existing teacher PCA for consistency)")
        else:
            # Dimensions mismatch: skip PCA (already at native dimension)
            print(f"⚠ Teacher PCA expects {pca_input_dim}D input, but got {input_dim}D")
            print(f"  Skipping PCA: using {modality} embeddings at native {input_dim}D dimension")
            print(f"  (This is expected for cell-only/gene-only ablations)")
        
        # Store reference to teacher's PCA model (if used)
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        pca_path = output_dir / "preprocessing" / "student_pca_model.pkl"
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
    elif pca_var is not None and input_dim >= 1024:
        # Only fit new PCA if input is high-dimensional (e.g., 1024D UNI)
        print(f"⚠ Teacher PCA model not found at {teacher_pca_path}")
        print(f"  Input is high-dimensional ({input_dim}D), fitting new PCA with variance={pca_var}")
        pca = PCA(n_components=pca_var)
        embeddings_array = pca.fit_transform(embeddings_array)
        print(f"✓ Applied new PCA: {input_dim}D → {embeddings_array.shape[1]}D")
        
        # Save fallback PCA model
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        pca_path = output_dir / "preprocessing" / "student_pca_model.pkl"
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
    else:
        # Low-dimensional input: use as-is
        if pca_var is not None:
            print(f"⚠ PCA variance set to {pca_var} but input is only {input_dim}D (already low-dim)")
        print(f"✓ Using {modality} embeddings at native dimension: {input_dim}D")
    
    # Rename for clarity - this now holds whatever embedding we loaded (UNI, scVI, or RCTD)
    X_student = embeddings_array
    
    # Load teacher preprocessed embeddings (full multimodal for getting teacher outputs)
    teacher_fused_path = teacher_preproc_dir / "fused_embeddings_pca.npy"
    
    if teacher_fused_path.exists():
        X_for_teacher = np.load(teacher_fused_path).astype(np.float32)
        print(f"✓ Loaded teacher preprocessed embeddings: {X_for_teacher.shape}")
        
        # CRITICAL: Align student samples with teacher's filtered samples
        # The teacher preprocessing may have removed rows with NaNs or issues
        # First, establish a safe bound for all arrays
        max_teacher_samples = X_for_teacher.shape[0]  # 23342
        max_student_samples = X_student.shape[0]      # 23395 for raw embeddings
        max_label_samples = len(labels)               # 23342 (pre-filtered to teacher)
        
        # Use the minimum size as safe bound to avoid index errors
        safe_size = min(max_student_samples, max_label_samples, max_teacher_samples)
        print(f"  Array sizes: Student={max_student_samples}, Labels={max_label_samples}, Teacher={max_teacher_samples}")
        print(f"  Using safe size: {safe_size} (minimum of all three)")
        
        # First, truncate all arrays to the safe size
        if max_student_samples > safe_size:
            X_student = X_student[:safe_size]
            barcode_indices = barcode_indices[:safe_size]
        if max_label_samples > safe_size:
            labels = labels[:safe_size]
        
        # Now try barcode matching within the safe bounds
        teacher_barcodes_path = teacher_preproc_dir / "barcodes.npy"
        if teacher_barcodes_path.exists():
            teacher_barcodes = np.load(teacher_barcodes_path, allow_pickle=True)
            print(f"✓ Loaded teacher's filtered barcodes: {teacher_barcodes.shape[0]} samples")
            
            # Find which student samples match teacher's filtered samples
            # Convert to strings for comparison (safer for barcode matching)
            student_barcode_str = barcode_indices.astype(str)
            teacher_barcode_str = teacher_barcodes.astype(str)
            
            # Find indices where student barcodes match teacher barcodes
            matching_indices = []
            teacher_barcode_set = set(teacher_barcode_str)
            for idx, barcode in enumerate(student_barcode_str):
                if barcode in teacher_barcode_set:
                    matching_indices.append(idx)
            
            if len(matching_indices) > 0 and len(matching_indices) == safe_size:
                # All student barcodes match teacher (perfect alignment)
                print(f"✓ Perfect barcode alignment: {len(matching_indices)} samples match exactly")
                print(f"  Student samples: {len(matching_indices)}")
                print(f"  Teacher samples: {X_for_teacher.shape[0]}")
            elif len(matching_indices) > 0:
                # Partial match - some student barcodes don't match teacher
                print(f"⚠ Partial barcode alignment: {len(matching_indices)}/{safe_size} samples match")
                print(f"  Filtering to matching indices only")
                # Align all arrays to matching indices
                X_student = X_student[matching_indices]
                barcode_indices = barcode_indices[matching_indices]
                labels = labels[matching_indices]
            else:
                print(f"⚠ Warning: No matching barcodes found - using simple truncation")
        else:
            print(f"⚠ Warning: Teacher barcodes not found - using simple truncation")
            
        # Final verification
        if X_student.shape[0] == len(labels) == X_for_teacher.shape[0]:
            print(f"✓ Final alignment verified: all arrays have {X_student.shape[0]} samples")
        else:
            print(f"⚠ Alignment warning: Student={X_student.shape[0]}, Labels={len(labels)}, Teacher={X_for_teacher.shape[0]}")
    else:
        print(f"⚠ Warning: Teacher embeddings not found at {teacher_fused_path}")
        X_for_teacher = None
    
    return X_student, X_for_teacher, labels, barcode_indices


def load_teacher_model(config: Dict, device='cpu'):
    """Load ensemble teacher model with dynamic input dimensions"""
    
    print("\n[2] Loading teacher model...")
    
    teacher_path = Path(config['distillation']['teacher_model_path'])
    
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher model not found at: {teacher_path}\n"
            f"Expected teacher to be trained first. Ensure unified pipeline was run correctly."
        )
    
    # Load the model/state dict first to determine dimensions
    loaded = torch.load(teacher_path, map_location=device)
    
    # Determine input dimension from loaded state dict
    if isinstance(loaded, dict):
        # State dict - extract from first layer weight
        first_layer_weight = loaded.get('network.0.weight', None)
        if first_layer_weight is None:
            first_layer_weight = loaded.get('fc1.weight', None)
        if first_layer_weight is not None:
            input_dim = first_layer_weight.shape[1]
        else:
            input_dim = 163  # Default to GEO
    else:
        # Full model object - get from layer
        input_dim = loaded.network[0].in_features if hasattr(loaded, 'network') else 163
    
    # Create teacher model with correct dimensions
    teacher = EcotypeClassifier(input_dim=input_dim, hidden_dims=[256, 128, 64], 
                              num_classes=5, dropout=0.3)
    teacher = teacher.to(device)
    
    # Load the state dict
    if isinstance(loaded, dict):
        teacher.load_state_dict(loaded)
    else:
        if hasattr(loaded, 'state_dict'):
            teacher.load_state_dict(loaded.state_dict())
        else:
            teacher = loaded.to(device)
    
    teacher.eval()
    
    # Freeze
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded teacher from {teacher_path}")
    print(f"  Input dimension: {input_dim}D")
    print(f"  Frozen parameters: {sum(1 for p in teacher.parameters() if not p.requires_grad)} trainable=0")
    
    return teacher


def get_teacher_outputs(teacher, X_teacher, device='cpu', batch_size=32):
    """Get teacher predictions for all data"""
    
    if X_teacher is None:
        raise ValueError("Teacher embeddings required")
    
    teacher.eval()
    X_tensor = torch.FloatTensor(X_teacher).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logits_list = []
    with torch.no_grad():
        for X_batch, in loader:
            logits = teacher(X_batch)
            logits_list.append(logits.cpu().numpy())
    
    return np.vstack(logits_list)


def generate_visualizations(fold_results, all_true, all_preds, label_encoder, output_dir):
    """Generate comprehensive visualizations for student training results"""
    
    visualizations_dir = Path(output_dir) / "training" / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[5] Generating Visualizations")
    print("─" * 80)
    
    # Extract metrics for plotting
    accuracies = [r['accuracy'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1_scores_list = [r['f1'] for r in fold_results]
    
    # Calculate per-class accuracy
    per_class_accs = []
    for class_idx in range(len(label_encoder.classes_)):
        class_mask = all_true == class_idx
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == class_idx).mean()
            per_class_accs.append(float(class_acc))
        else:
            per_class_accs.append(0.0)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Student: Confusion Matrix (5-Fold CV)')
    plt.tight_layout()
    plt.savefig(visualizations_dir / '01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 01_confusion_matrix.png")
    
    # 2. Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fold_idx, result in enumerate(fold_results):
        history = result['history']
        axes[0].plot(history['train_loss'], label=f"Fold {fold_idx+1} (train)", alpha=0.7)
        axes[1].plot(history['train_acc'], label=f"Fold {fold_idx+1} (train)", alpha=0.7)
        axes[0].plot(history['val_loss'], '--', label=f"Fold {fold_idx+1} (val)", alpha=0.7)
        axes[1].plot(history['val_acc'], '--', label=f"Fold {fold_idx+1} (val)", alpha=0.7)
        
        # Mark early stopping point
        early_stop_epoch = len(history['val_loss']) - 1
        axes[0].plot(early_stop_epoch, history['val_loss'][early_stop_epoch], 
                    marker='*', markersize=20, color='red', markeredgecolor='darkred', 
                    markeredgewidth=1.5, zorder=5)
        axes[1].plot(early_stop_epoch, history['val_acc'][early_stop_epoch], 
                    marker='*', markersize=20, color='red', markeredgecolor='darkred', 
                    markeredgewidth=1.5, zorder=5)
    
    early_stop_element = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                markeredgecolor='darkred', markeredgewidth=1.5, markersize=15, 
                                label='Early Stopping', linestyle='')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Student: Training & Validation Loss')
    handles_0, labels_0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles_0 + [early_stop_element], labels_0 + ['Early Stopping'], 
                  fontsize=8, ncol=2, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Student: Training & Validation Accuracy')
    handles_1, labels_1 = axes[1].get_legend_handles_labels()
    axes[1].legend(handles_1 + [early_stop_element], labels_1 + ['Early Stopping'], 
                  fontsize=8, ncol=2, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visualizations_dir / '02_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 02_training_curves.png")
    
    # 3. Per-Class Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(label_encoder.classes_))
    width = 0.15
    
    for fold_idx, result in enumerate(fold_results):
        offset = (fold_idx - 2) * width
        ax.bar(x_pos + offset, result['per_class_accuracy'], width, 
               label=f"Fold {fold_idx+1}", alpha=0.8)
    
    ax.plot(x_pos, per_class_accs, 'k-', marker='o', linewidth=3, 
            markersize=8, label='Average', zorder=10)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Ecotype Class')
    ax.set_title('Student: Per-Class Accuracy (5-Fold CV)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(label_encoder.classes_, rotation=15, ha='right')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(visualizations_dir / '03_per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 03_per_class_accuracy.png")
    
    # 4. Metrics Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_means = [np.mean(accuracies), np.mean(precisions), 
                     np.mean(recalls), np.mean(f1_scores_list)]
    metrics_stds = [np.std(accuracies), np.std(precisions), 
                    np.std(recalls), np.std(f1_scores_list)]
    
    x_pos = np.arange(len(metrics_names))
    ax.bar(x_pos, metrics_means, yerr=metrics_stds, capsize=5, 
           alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    ax.set_ylabel('Score')
    ax.set_title('Student: Overall Metrics (Mean ± Std, 5-Fold CV)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(metrics_means, metrics_stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(visualizations_dir / '04_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 04_metrics_summary.png")
    
    # 5. Fold Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    fold_nums = [r['fold'] for r in fold_results]
    x_pos = np.arange(len(fold_nums))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x_pos - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x_pos + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x_pos + 1.5*width, f1_scores_list, width, label='F1 Score', alpha=0.8)
    
    ax.axhline(y=np.mean(accuracies), color='C0', linestyle='--', alpha=0.5, label='Acc Mean')
    ax.axhline(y=np.mean(f1_scores_list), color='C3', linestyle='--', alpha=0.5, label='F1 Mean')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Fold')
    ax.set_title('Student: Metrics per Fold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i}' for i in fold_nums])
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(visualizations_dir / '05_fold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 05_fold_comparison.png")
    
    print(f"\n✓ Visualizations saved to {visualizations_dir}")


def train_student(config: Dict, teacher_model, X_student, X_teacher, y, barcode_indices=None, device='cpu', output_dir=None):
    """Train student model with knowledge distillation"""
    
    if output_dir is None:
        output_dir = Path(config['output']['output_dir'])
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Get teacher outputs for entire dataset
    print("\n[3] Getting teacher outputs...")
    teacher_logits_all = get_teacher_outputs(teacher_model, X_teacher, device=device)
    print(f"✓ Teacher outputs: {teacher_logits_all.shape}")
    
    # Setup CV with proper random_seed handling
    random_seed = config.get('pipeline', {}).get('random_seed', 
                            config.get('training', {}).get('random_seed', 42))
    skf = StratifiedKFold(n_splits=config['training']['n_folds'], 
                         shuffle=True, 
                         random_state=random_seed)
    
    fold_results = []
    fold_models = []
    all_preds = []
    all_true = []
    fold_histories = {}
    
    print("\n[4] Starting 5-fold cross-validation student training...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_student, y_encoded)):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/5")
        print(f"{'='*70}")
        
        # Split data
        X_train, X_val = X_student[train_idx], X_student[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        teacher_logits_train = teacher_logits_all[train_idx]
        teacher_logits_val = teacher_logits_all[val_idx]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train),
            torch.FloatTensor(teacher_logits_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
            torch.FloatTensor(teacher_logits_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=int(config['training']['batch_size']), 
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(config['training']['batch_size']), 
                               shuffle=False)
        
        # Create student model
        input_dim = X_train.shape[1]
        student = StudentMLPClassifier(
            input_dim=input_dim,
            hidden_dims=config['training']['hidden_dims'],
            n_classes=len(le.classes_),
            dropout_rate=float(config['training']['dropout_rate'])
        ).to(device)
        
        # Setup optimization
        optimizer = optim.Adam(student.parameters(),
                             lr=float(config['training']['learning_rate']),
                             weight_decay=float(config['training']['weight_decay']))
        
        criterion = KnowledgeDistillationLoss(
            alpha=float(config['distillation']['alpha']),
            beta=float(config['distillation']['beta']),
            gamma=float(config['distillation']['gamma']),
            temperature=float(config['distillation']['temperature'])
        )
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(int(config['training']['n_epochs'])):
            # Train
            student.train()
            train_loss = 0
            for X_batch, y_batch, t_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                t_batch = t_batch.to(device)
                
                optimizer.zero_grad()
                student_logits = student(X_batch)
                student_features = student.network[:-1](X_batch)  # Get penultimate layer
                
                loss, loss_dict = criterion(student_logits, t_batch, student_features, 
                                           t_batch, y_batch)  # Note: using teacher features placeholder
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            student.eval()
            val_loss = 0
            train_acc = 0
            val_acc = 0
            
            with torch.no_grad():
                # Train accuracy
                for X_batch, y_batch, _ in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = torch.argmax(student(X_batch), dim=1)
                    train_acc += (preds == y_batch).sum().item()
                train_acc /= len(y_train)
                
                # Val accuracy & loss
                val_preds_list = []
                for X_batch, y_batch, t_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    t_batch = t_batch.to(device)
                    
                    student_logits = student(X_batch)
                    student_features = student.network[:-1](X_batch)
                    loss, _ = criterion(student_logits, t_batch, student_features, 
                                       t_batch, y_batch)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(student_logits, dim=1).cpu().numpy()
                    val_preds_list.append(preds)
                
                val_loss /= len(val_loader)
                val_preds = np.concatenate(val_preds_list)
                val_acc = accuracy_score(y_val, val_preds)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                models_dir = output_dir / "training" / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), 
                          models_dir / f"fold_{fold_idx}_best_student_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= int(config['training']['early_stopping_patience']):
                print(f"  Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
        
        fold_histories[f"fold_{fold_idx}"] = history
        
        # Get final predictions on validation set for per-class accuracy
        student.eval()
        fold_preds_list = []
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                preds = torch.argmax(student(X_batch), dim=1).cpu().numpy()
                fold_preds_list.append(preds)
        
        fold_preds = np.concatenate(fold_preds_list) if fold_preds_list else np.array([])
        
        # Compute per-class accuracy for this fold
        per_class_acc = []
        for class_id in range(len(le.classes_)):
            class_mask = y_val == class_id
            if class_mask.sum() > 0 and len(fold_preds) > 0:
                class_acc = (fold_preds[class_mask] == class_id).mean()
                per_class_acc.append(float(class_acc))
            else:
                per_class_acc.append(0.0)
        
        # Calculate fold metrics
        fold_acc = accuracy_score(y_val, fold_preds)
        fold_prec = precision_score(y_val, fold_preds, average='weighted', zero_division=0)
        fold_rec = recall_score(y_val, fold_preds, average='weighted', zero_division=0)
        fold_f1 = f1_score(y_val, fold_preds, average='weighted', zero_division=0)
        
        # Add metrics to history for visualization
        history['accuracy'] = fold_acc
        history['precision'] = fold_prec
        history['recall'] = fold_rec
        history['f1'] = fold_f1
        history['per_class_accuracy'] = per_class_acc
        
        # Collect fold predictions and labels for overall metrics
        all_preds.extend(fold_preds)
        all_true.extend(y_val)
        fold_models.append(student.state_dict())
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'precision': fold_prec,
            'recall': fold_rec,
            'f1': fold_f1,
            'n_train': len(train_idx),
            'n_val': len(val_idx)
        })
        
        print(f"✓ Fold {fold_idx+1} - Acc: {fold_acc:.4f}, Prec: {fold_prec:.4f}, "
              f"Rec: {fold_rec:.4f}, F1: {fold_f1:.4f}")
    
    # Overall metrics
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    
    overall_acc = accuracy_score(all_true, all_preds)
    overall_prec = precision_score(all_true, all_preds, average='weighted')
    overall_rec = recall_score(all_true, all_preds, average='weighted')
    overall_f1 = f1_score(all_true, all_preds, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS (5-Fold CV)")
    print(f"{'='*70}")
    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"Precision: {overall_prec:.4f}")
    print(f"Recall:    {overall_rec:.4f}")
    print(f"F1 Score:  {overall_f1:.4f}")
    
    # Save results
    metrics_dir = output_dir / "training" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(v) for v in obj]
        return obj
    
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_samples': int(len(y)),  # Ensure native int
            'n_classes': int(len(le.classes_)),  # Ensure native int
            'class_names': convert_to_native(list(le.classes_))  # Convert to list
        },
        'overall_metrics': {
            'accuracy': float(overall_acc),
            'precision': float(overall_prec),
            'recall': float(overall_rec),
            'f1': float(overall_f1)
        },
        'per_fold': [convert_to_native(f) for f in fold_results]
    }
    
    with open(metrics_dir / "student_training_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save predictions WITH ACTUAL BARCODES for proper patient linking
    predictions_data = {
        'barcode': barcode_indices if barcode_indices is not None else range(len(all_true)),
        'true_label': le.inverse_transform(all_true),
        'predicted_label': le.inverse_transform(all_preds),
        'correct': all_true == all_preds
    }
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(metrics_dir / "student_predictions.csv", index=False)
    
    # Save label encoder
    with open(metrics_dir / "student_label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\n✓ Results saved to {metrics_dir}")
    
    return results_json, fold_histories, all_true, all_preds, le


def main():
    parser = argparse.ArgumentParser(description="Unified student training with distillation")
    parser.add_argument('--config', type=str, required=True, help='Student config YAML')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Type conversions
    config['training']['learning_rate'] = float(config['training'].get('learning_rate', 0.001))
    config['training']['batch_size'] = int(config['training'].get('batch_size', 32))
    config['training']['n_epochs'] = int(config['training'].get('n_epochs', 150))
    config['training']['n_folds'] = int(config['training'].get('n_folds', 5))
    config['distillation']['alpha'] = float(config['distillation'].get('alpha', 0.7))
    config['distillation']['beta'] = float(config['distillation'].get('beta', 0.2))
    config['distillation']['gamma'] = float(config['distillation'].get('gamma', 0.1))
    config['distillation']['temperature'] = float(config['distillation'].get('temperature', 4.0))
    
    output_dir = Path(config['output']['output_dir'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("UNIFIED STUDENT MODEL TRAINING WITH KNOWLEDGE DISTILLATION")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    try:
        # Load all data (including barcode indices for prediction tracking)
        X_student, X_teacher, y, barcode_indices = load_data(config)
        
        # Load teacher
        teacher = load_teacher_model(config, device=device)
        
        # Train student
        results, histories, all_true, all_preds, le = train_student(
            config, teacher, X_student, X_teacher, y, 
            barcode_indices=barcode_indices,
            device=device, output_dir=output_dir
        )
        
        # Generate visualizations
        # Extract fold results from histories (histories is a dict of fold_idx -> history_dict)
        fold_results_for_viz = []
        for fold_key, history in histories.items():
            fold_idx = int(fold_key.split('_')[1]) + 1  # Extract fold number from "fold_0", "fold_1", etc.
            fold_results_for_viz.append({
                'fold': fold_idx,
                'accuracy': history.get('accuracy', 0),
                'precision': history.get('precision', 0),
                'recall': history.get('recall', 0),
                'f1': history.get('f1', 0),
                'per_class_accuracy': history.get('per_class_accuracy', []),
                'history': {
                    'train_loss': history.get('train_loss', []),
                    'val_loss': history.get('val_loss', []),
                    'train_acc': history.get('train_acc', []),
                    'val_acc': history.get('val_acc', [])
                }
            })
        
        generate_visualizations(fold_results_for_viz, all_true, all_preds, le, output_dir)
        
        # ════════════════════════════════════════════════════════════════════
        # COMPREHENSIVE STUDENT VISUALIZATIONS
        # ════════════════════════════════════════════════════════════════════
        
        if VISUALIZATION_AVAILABLE:
            logger.info("\n" + "="*80)
            logger.info("GENERATING COMPREHENSIVE STUDENT VISUALIZATIONS")
            logger.info("="*80)
            
            try:
                # Get teacher predictions for all data
                teacher_logits = get_teacher_outputs(teacher, X_teacher, device=device)
                teacher_preds = np.argmax(teacher_logits, axis=1)
                teacher_confidence = np.max(F.softmax(torch.FloatTensor(teacher_logits), dim=1).numpy(), axis=1)
                
                # Get student predictions on all data
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=config['training']['n_folds'], 
                                     shuffle=True, random_state=42)
                
                y_encoded = le.fit_transform(y) if hasattr(le, 'fit_transform') else le.transform(y)
                student_all_probs = []
                student_all_preds = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_student, y_encoded)):
                    X_val = X_student[val_idx]
                    
                    # Load best student model for this fold
                    models_dir = output_dir / "training" / "models"
                    model_path = models_dir / f"fold_{fold_idx}_best_student_model.pth"
                    
                    if model_path.exists():
                        student_model = StudentMLPClassifier(
                            input_dim=X_student.shape[1],
                            hidden_dims=config['training']['hidden_dims'],
                            n_classes=len(le.classes_),
                            dropout_rate=float(config['training']['dropout_rate'])
                        ).to(device)
                        student_model.load_state_dict(torch.load(model_path))
                        student_model.eval()
                        
                        with torch.no_grad():
                            X_val_tensor = torch.FloatTensor(X_val).to(device)
                            logits = student_model(X_val_tensor)
                            probs = F.softmax(logits, dim=1).cpu().numpy()
                            preds = np.argmax(probs, axis=1)
                        
                        student_all_probs.append(probs)
                        student_all_preds.append(preds)
                
                # Merge predictions (handle overlaps by averaging probabilities)
                student_probs_all = np.zeros((len(X_student), len(le.classes_)))
                student_preds_all = np.zeros(len(X_student), dtype=int)
                fold_counts = np.zeros(len(X_student))
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_student, y_encoded)):
                    if fold_idx < len(student_all_probs):
                        student_probs_all[val_idx] += student_all_probs[fold_idx]
                        fold_counts[val_idx] += 1
                
                # Average probabilities
                for i in range(len(X_student)):
                    if fold_counts[i] > 0:
                        student_probs_all[i] /= fold_counts[i]
                        student_preds_all[i] = np.argmax(student_probs_all[i])
                
                student_confidence = np.max(student_probs_all, axis=1)
                
                # Create predictions dataframe with ACTUAL BARCODES
                predictions_df = pd.DataFrame({
                    'barcode': barcode_indices,
                    'prediction': student_preds_all.astype(int),
                    'confidence': student_confidence.astype(float),
                    'teacher_pred': teacher_preds.astype(int),
                    'teacher_confidence': teacher_confidence.astype(float)
                })
                
                # Load metadata if available
                metadata_df = None
                metadata_path = Path(config['input_dataset'].get('metadata_file', ''))
                if metadata_path.exists():
                    metadata_df = pd.read_csv(metadata_path)
                    # Ensure barcode is a column, not index
                    if 'barcode' not in metadata_df.columns and metadata_df.index.name == 'barcode':
                        metadata_df = metadata_df.reset_index()
                    # Add patient column if only patient_id exists
                    if 'patient_id' in metadata_df.columns and 'patient' not in metadata_df.columns:
                        metadata_df['patient'] = metadata_df['patient_id']
                    logger.info(f"  ✓ Loaded metadata: {len(metadata_df)} samples")
                
                # Generate visualizations
                viz_output = output_dir / "post-training" / "visualizations"
                generate_all_student_visualizations(
                    predictions_df=predictions_df,
                    metadata_df=metadata_df,
                    embeddings=X_student,
                    labels=y_encoded,
                    output_dir=str(viz_output),
                    config=config,
                    label_encoder=le,
                    pca_variance=config['embeddings'].get('image_encoder', {}).get('pca_variance', 0.95)
                )
                
                logger.info(f"\n✅ Comprehensive visualizations saved to:")
                logger.info(f"   {viz_output}")
                
            except Exception as viz_error:
                logger.warning(f"⚠ Error during visualization generation: {viz_error}")
                import traceback
                traceback.print_exc()
        
        
        print("\n" + "=" * 80)
        print("✅ STUDENT TRAINING COMPLETE")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
