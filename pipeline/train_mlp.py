#!/usr/bin/env python3
"""
Phase 3: MLP Training Pipeline with 5-Fold Cross-Validation
=============================================================

Trains ecotype classifier on fused embeddings with comprehensive metrics:
- 5-fold stratified cross-validation
- Class weight balancing
- Early stopping + model selection
- Per-class accuracy and confusion matrices
- Detailed training visualization

Usage:
    python pipeline/train_mlp.py \
        --embeddings data/preprocessed_arrays/fused_embeddings_pca.npy \
        --labels data/input_dataset/barcode_labels.csv \
        --output results

Or with defaults:
    python pipeline/train_mlp.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

# Set random seeds for reproducibility
def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: ensure deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Set seed at module import
set_random_seed(42)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Import metrics and visualization modules
from metrics_tracker import MetricsTracker
from post_training_visualizations import PostTrainingVisualizer

print("\n" + "="*100)
print(" PHASE 3: MLP TRAINING WITH 5-FOLD CROSS-VALIDATION")
print("="*100 + "\n")


# ============================================================================
# MLP CLASSIFIER
# ============================================================================

class EcotypeClassifier(nn.Module):
    """Multi-layer perceptron for ecotype classification"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # Matching old architecture
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with batch norm and dropout
        # NOTE: Order is Linear→ReLU→BatchNorm→Dropout (matches old working code)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# TRAINING
# ============================================================================

class MLPTrainer:
    """MLP training with early stopping and class weight balancing"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 learning_rate: float = 1e-3, batch_size: int = 32,
                 patience: int = 20):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.patience = patience
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.best_model_state = None
    
    def _get_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Calculate inverse frequency weights for class balancing"""
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts)
        weights = weights / weights.sum() * len(classes)  # normalize
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).long()
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == y_batch).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
        
        return total_loss / n_batches, total_acc / n_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).long()
            
            logits = self.model(X_batch)
            loss = criterion(logits, y_batch)
            
            preds = logits.argmax(dim=1)
            acc = (preds == y_batch).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
        
        return total_loss / n_batches, total_acc / n_batches
    
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        logits = self.model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1)[0].cpu().numpy()
        
        return preds, confs
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 200, verbose: bool = True,
            metrics_tracker=None, fold_idx: int = 0):
        """Train with early stopping"""
        
        # Move model to device BEFORE training
        self.model = self.model.to(self.device)
        
        # Create data loaders
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Class weights for imbalanced data
        class_weights = self._get_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping on validation loss
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                is_best = True
                # Save best model state
                self.best_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Log to metrics tracker if provided
            if metrics_tracker is not None:
                metrics_tracker.log_epoch(
                    fold=fold_idx,
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    is_best=is_best,
                    early_stop=(patience_counter >= self.patience)
                )
            
            if patience_counter >= self.patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_data(embeddings_path: str, labels_path: str, metadata_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load fused embeddings, labels, barcodes, and spatial metadata"""
    
    print("[STAGE 1] Loading Data")
    print("─" * 100)
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"  ✓ Embeddings: {embeddings.shape}")
    
    # Load barcodes that were saved during preprocessing  (CRITICAL - determines ordering!)
    embeddings_dir = Path(embeddings_path).parent
    barcodes_path = embeddings_dir / 'barcodes.npy'
    
    if barcodes_path.exists():
        preprocessing_barcodes = np.load(barcodes_path, allow_pickle=True)
        print(f"  ✓ Preprocessing barcodes loaded: {len(preprocessing_barcodes)} samples")
    else:
        print(f"  ⚠ WARNING: Barcodes file not found at {barcodes_path}")
        print(f"             Will try to load from labels CSV (may cause misalignment!)")
        preprocessing_barcodes = None
    
    # Load labels from CSV
    labels_df = pd.read_csv(labels_path, index_col=0)
    print(f"  ✓ Labels file loaded: {len(labels_df)} rows")
    
    # CRITICAL FIX: Reindex labels to match preprocessing barcode order
    if preprocessing_barcodes is not None:
        # Reindex labels_df to match the preprocessing barcodes order
        labels_df = labels_df.reindex(preprocessing_barcodes)
        
        if labels_df.isna().any().any():
            print(f"  ⚠ WARNING: Some barcodes from preprocessing not found in labels!")
            # Drop NaN rows
            labels_df = labels_df.dropna()
            print(f"  ✓ Kept {len(labels_df)} matching labels")
        
        barcodes = preprocessing_barcodes
    else:
        # Fallback: use barcodes from CSV (order may not match embeddings)
        barcodes = labels_df.index.values
    
    # Extract labels in the correct order
    labels = labels_df.iloc[:, 0].values
    
    print(f"  ✓ Labels: {labels.shape}")
    print(f"  ✓ Barcodes: {len(barcodes)}")
    
    # Load spatial metadata if available
    spatial_data = None
    if metadata_path and Path(metadata_path).exists():
        try:
            spatial_data = pd.read_csv(metadata_path)
            # Match barcodes with spatial data
            spatial_data['barcode_short'] = spatial_data['original_barcode']
            spatial_map = dict(zip(spatial_data['original_barcode'], spatial_data[['patient_id', 'x_coord', 'y_coord']]))
            print(f"  ✓ Spatial metadata: {len(spatial_data)} spots")
        except Exception as e:
            print(f"  ⚠ Could not load spatial metadata: {e}")
            spatial_data = None
    else:
        print(f"  ⚠ Spatial metadata not found at {metadata_path}")
    
    # Check alignment
    if len(embeddings) != len(labels):
        print(f"  ⚠ Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
        min_len = min(len(embeddings), len(labels))
        embeddings = embeddings[:min_len]
        labels = labels[:min_len]
        barcodes = barcodes[:min_len]
        print(f"  ✓ Aligned to: {min_len} samples")
    
    return embeddings, labels, barcodes, spatial_data


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: MLP training with 5-fold cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='Path to config.yaml file (optional - contains training parameters)'
    )
    
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to fused embeddings'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=False,
        help='Path to labels CSV (optional if config provided)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Maximum epochs (default: from config or 200)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: from config or 32)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (default: from config or 1e-3)'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        training_config = config.get('training', {})
        
        # Get labels from config if not provided via args
        if not args.labels:
            input_dataset_config = config.get('input_dataset', {})
            labels_file = input_dataset_config.get('labels_file', '')
            if labels_file:
                labels_path = Path(labels_file)
                if not labels_path.is_absolute():
                    labels_path = Path.cwd() / labels_file
                args.labels = str(labels_path)
        
        # Use config values as defaults if not provided via command line
        if args.epochs is None:
            args.epochs = training_config.get('n_epochs', 200)
        if args.batch_size is None:
            args.batch_size = training_config.get('batch_size', 32)
        if args.learning_rate is None:
            args.learning_rate = training_config.get('learning_rate', 1e-3)
    else:
        # Use defaults if no config
        if args.epochs is None:
            args.epochs = 200
        if args.batch_size is None:
            args.batch_size = 32
        if args.learning_rate is None:
            args.learning_rate = 1e-3
        if not args.labels:
            print("  ✗ ERROR: Either --config or --labels must be provided")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organized output
    metrics_dir = output_dir / 'metrics'
    visualizations_dir = output_dir / 'visualizations'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Embeddings:  {args.embeddings}")
    print(f"Labels:      {args.labels}")
    print(f"Output:      {output_dir}\n")
    
    # Load data with spatial metadata
    metadata_path = Path(args.labels).parent / 'barcode_metadata.csv'
    X, y, barcodes, spatial_data = load_data(args.embeddings, args.labels, str(metadata_path))
    print()
    
    # Encode labels
    print("[STAGE 2] Label Encoding")
    print("─" * 100)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"  Distribution: {dict(zip(label_encoder.classes_, np.bincount(y_encoded)))}")
    print()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        save_dir=str(metrics_dir)
    )
    metrics_tracker.config = {
        'n_folds': 5,
        'embedding_dim': X.shape[1],
        'n_classes': len(label_encoder.classes_),
        'class_names': list(label_encoder.classes_),
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.epochs,
        'patience': 20
    }
    print("[STAGE 2.5] Metrics Tracker Initialized")
    print(f"  ✓ Output directory: {metrics_tracker.save_dir}")
    print()
    
    # 5-Fold Cross-Validation
    print("[STAGE 3] 5-Fold Stratified Cross-Validation")
    print("─" * 100)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_probs = []
    all_y_confs = []  # Collect confidence scores from each fold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print(f"  ⚠ WARNING: CUDA not available, using CPU\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"  Fold {fold_idx + 1}/5", flush=True)
        
        # Reset random seed for each fold to ensure reproducibility
        set_random_seed(42 + fold_idx)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Create model
        model = EcotypeClassifier(
            input_dim=X.shape[1],
            num_classes=len(label_encoder.classes_),
            hidden_dims=[256, 128, 64],  # Matching old working architecture
            dropout=0.3
        )
        
        trainer = MLPTrainer(
            model,
            device=device,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=20  # Matching old working patience
        )
        
        # Train
        trainer.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, verbose=False,
                   metrics_tracker=metrics_tracker, fold_idx=fold_idx)
        
        # Predict on validation set
        y_pred, y_confs = trainer.predict(X_val)
        
        # Get probabilities for ROC-AUC (all classes)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            model.eval()
            logits = model(X_val_tensor)
            y_probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_acc = []
        for class_idx in range(len(label_encoder.classes_)):
            class_mask = y_val == class_idx
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == class_idx).mean()
                per_class_acc.append(float(class_acc))
            else:
                per_class_acc.append(0.0)
        
        # Log fold completion to metrics tracker
        metrics_tracker.log_fold_complete(
            fold=fold_idx,
            accuracy=float(acc),
            precision=float(prec),
            recall=float(rec),
            f1_score=float(f1),
            per_class_accuracy=per_class_acc
        )
        
        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'per_class_accuracy': per_class_acc,
            'model_state': model.state_dict(),
            'trainer': trainer
        }
        
        fold_results.append(fold_result)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_confs.extend(y_confs)  # Collect confidence scores for each prediction
        all_y_probs.append(y_probs)
        
        print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_confs = np.array(all_y_confs)  # Convert confidence scores to numpy array
    all_y_probs = np.vstack(all_y_probs)
    
    print()
    
    # Aggregate metrics
    print("[STAGE 4] Cross-Validation Results")
    print("─" * 100)
    
    accuracies = [r['accuracy'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1_scores_list = [r['f1'] for r in fold_results]
    
    print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1 Score:  {np.mean(f1_scores_list):.4f} ± {np.std(f1_scores_list):.4f}")
    print()
    
    # Per-class accuracy across folds
    print("  Per-Class Accuracy (avg across folds):")
    per_class_accs = np.array([r['per_class_accuracy'] for r in fold_results]).mean(axis=0)
    for class_name, class_acc in zip(label_encoder.classes_, per_class_accs):
        print(f"    {str(class_name):25s}: {class_acc:.4f}")
    print()
    
    # Save results
    print("[STAGE 5] Saving Results")
    print("─" * 100)
    
    # Convert numpy types to Python native types for JSON serialization
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(label_encoder.classes_)),
            'classes': [int(c) for c in label_encoder.classes_],
        },
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'patience': 20,
            'hidden_dims': [256, 128, 64],
            'dropout': 0.3,
        },
        'metrics': {
            'overall': {
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'precision_mean': float(np.mean(precisions)),
                'precision_std': float(np.std(precisions)),
                'recall_mean': float(np.mean(recalls)),
                'recall_std': float(np.std(recalls)),
                'f1_mean': float(np.mean(f1_scores_list)),
                'f1_std': float(np.std(f1_scores_list)),
            },
            'per_fold': [
                {
                    'fold': int(r['fold']),
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1': float(r['f1']),
                    'per_class_accuracy': [float(x) for x in r['per_class_accuracy']]
                }
                for r in fold_results
            ],
            'per_class_avg_accuracy': {
                str(int(class_name)): float(class_acc)
                for class_name, class_acc in zip(label_encoder.classes_, per_class_accs)
            }
        }
    }
    
    # Save JSON results
    results_file = metrics_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"  ✓ training_results.json")
    
    # Save label encoder
    with open(metrics_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ label_encoder.pkl")
    
    # Save best model (fold 1)
    torch.save(fold_results[0]['model_state'], metrics_dir / 'model_best.pt')
    print(f"  ✓ model_best.pt (Fold 1)")
    
    print()
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print("[STAGE 6] Generating Visualizations")
    print("─" * 100)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (5-Fold CV)')
    plt.tight_layout()
    plt.savefig(visualizations_dir / '01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 01_confusion_matrix.png")
    
    # 2. Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training losses and accuracies
    for fold_idx, result in enumerate(fold_results):
        trainer = result['trainer']
        axes[0].plot(trainer.history['train_loss'], label=f"Fold {fold_idx+1} (train)", alpha=0.7)
        axes[1].plot(trainer.history['train_acc'], label=f"Fold {fold_idx+1} (train)", alpha=0.7)
    
    # Plot validation losses, accuracies, and mark early stopping points
    for fold_idx, result in enumerate(fold_results):
        trainer = result['trainer']
        val_loss_history = trainer.history['val_loss']
        val_acc_history = trainer.history['val_acc']
        
        # Plot validation curves
        axes[0].plot(val_loss_history, '--', label=f"Fold {fold_idx+1} (val)", alpha=0.7)
        axes[1].plot(val_acc_history, '--', label=f"Fold {fold_idx+1} (val)", alpha=0.7)
        
        # Mark early stopping point with red star
        early_stop_epoch = len(val_loss_history) - 1
        axes[0].plot(early_stop_epoch, val_loss_history[early_stop_epoch], 
                    marker='*', markersize=20, color='red', markeredgecolor='darkred', 
                    markeredgewidth=1.5, zorder=5)
        axes[1].plot(early_stop_epoch, val_acc_history[early_stop_epoch], 
                    marker='*', markersize=20, color='red', markeredgecolor='darkred', 
                    markeredgewidth=1.5, zorder=5)
    
    # Create custom legend element for early stopping marker
    early_stop_element = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                markeredgecolor='darkred', markeredgewidth=1.5, markersize=15, 
                                label='Early Stopping', linestyle='')
    
    # Configure left plot (Loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss (with Early Stopping Markers)')
    handles_0, labels_0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles_0 + [early_stop_element], labels_0 + ['Early Stopping'], 
                  fontsize=8, ncol=2, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Configure right plot (Accuracy)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy (with Early Stopping Markers)')
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
    
    # Overall average
    ax.plot(x_pos, per_class_accs, 'k-', marker='o', linewidth=3, 
            markersize=8, label='Average', zorder=10)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Ecotype Class')
    ax.set_title('Per-Class Accuracy (5-Fold CV)')
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
    ax.set_title('Overall Metrics (Mean ± Std, 5-Fold CV)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
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
    ax.set_title('Metrics per Fold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i}' for i in fold_nums])
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(visualizations_dir / '05_fold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 05_fold_comparison.png")
    
    # 6. Detailed Classification Report (Text)
    report_text = classification_report(
        all_y_true, all_y_pred,
        target_names=[str(c) for c in label_encoder.classes_],
        digits=4
    )
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT (5-FOLD CV)\n")
        f.write("="*80 + "\n\n")
        f.write(report_text)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("PER-CLASS ACCURACY BY FOLD\n")
        f.write("="*80 + "\n\n")
        
        for fold_idx, result in enumerate(fold_results):
            f.write(f"Fold {fold_idx + 1}:\n")
            for class_name, class_acc in zip(label_encoder.classes_, result['per_class_accuracy']):
                f.write(f"  {str(class_name):25s}: {class_acc:.4f}\n")
            f.write("\n")
    
    print(f"  ✓ classification_report.txt")
    
    print()
    
    # ========================================================================
    # METRICS EXPORT & ENHANCED VISUALIZATIONS
    # ========================================================================
    
    print("[STAGE 7] Exporting Metrics & Generating Enhanced Visualizations")
    print("─" * 100)
    
    # Export metrics to CSV/JSON
    metrics_tracker.export_all()
    print(f"  ✓ Metrics exported (training_metrics/)")
    
    # Create predictions DataFrame with spatial coordinates
    try:
        print("\n  Creating predictions DataFrame with spatial coordinates...")
        
        # Build mapping from barcode to spatial data
        spatial_map = {}
        if spatial_data is not None:
            for idx, row in spatial_data.iterrows():
                original_barcode = row['original_barcode']
                spatial_map[original_barcode] = {
                    'patient_id': row['patient_id'],
                    'x_coord': row['x_coord'],
                    'y_coord': row['y_coord']
                }
        
        # Extract spatial info for each spot
        patient_ids = []
        x_coords = []
        y_coords = []
        
        for barcode in barcodes:
            # Try to match with metadata
            if barcode in spatial_map:
                info = spatial_map[barcode]
                patient_ids.append(info['patient_id'])
                x_coords.append(info['x_coord'])
                y_coords.append(info['y_coord'])
            else:
                # Fallback: try to extract patient from barcode
                if '_' in str(barcode):
                    parts = str(barcode).split('_')
                    patient_id = parts[0]
                elif '-' in str(barcode):
                    patient_id = 'Unknown'
                else:
                    patient_id = 'Unknown'
                
                patient_ids.append(patient_id)
                x_coords.append(np.nan)
                y_coords.append(np.nan)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'barcode': barcodes,
            'patient_id': patient_ids,
            'x_coord': x_coords,
            'y_coord': y_coords,
            'predicted_label': all_y_pred,
            'predicted_ecotype': [label_encoder.classes_[p] for p in all_y_pred],
            'confidence': all_y_confs,  # Add confidence scores from model predictions
            'ground_truth_label': all_y_true,
            'ground_truth_ecotype': [label_encoder.classes_[t] for t in all_y_true]
        })
        
        # Save predictions DataFrame
        predictions_df.to_csv(metrics_dir / 'predictions_all_spots.csv', index=False)
        print(f"  ✓ predictions_all_spots.csv ({len(predictions_df)} spots)")
        
        # Count spatial data availability
        has_coords = (~predictions_df['x_coord'].isna()).sum()
        print(f"  ✓ Spots with spatial coordinates: {has_coords}/{len(predictions_df)}")
        
    except Exception as e:
        print(f"  ⚠ Error creating predictions DataFrame: {e}")
        predictions_df = None
    
    print()
    
    # ========================================================================
    # SPATIAL VISUALIZATIONS
    # ========================================================================
    
    if predictions_df is not None and has_coords > 0:
        print("[STAGE 8] Generating Spatial Visualizations")
        print("─" * 100)
        
        try:
            # Create visualizations directory
            viz_dir = output_dir / 'spatial_visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Get unique patients
            patients = predictions_df['patient_id'].unique()
            patients = sorted([p for p in patients if pd.notna(p)])
            
            print(f"  Found {len(patients)} patients: {', '.join(patients)}")
            
            # Color scheme for ecotypes
            ecotype_colors = {
                0: '#E74C3C',  # Red - Fibrotic
                1: '#3498DB',  # Blue - Immunosuppressive
                2: '#F39C12',  # Yellow/Orange - Invasive_Border
                3: '#2ECC71',  # Green - Metabolic
                4: '#9B59B6'   # Purple - Normal_Adjacent
            }
            
            ecotype_names_display = {
                0: 'Fibrotic',
                1: 'Immunosuppressive', 
                2: 'Invasive Border',
                3: 'Metabolic',
                4: 'Normal Adjacent'
            }
            
            # Generate per-patient spatial maps
            for patient_id in patients:
                patient_data = predictions_df[predictions_df['patient_id'] == patient_id].copy()
                patient_data = patient_data.dropna(subset=['x_coord', 'y_coord'])
                
                if len(patient_data) == 0:
                    continue
                
                print(f"\n  Generating visualizations for {patient_id}...")
                
                # 1. Ground Truth Spatial Map
                fig, ax = plt.subplots(figsize=(10, 10))
                
                scatter = ax.scatter(
                    patient_data['x_coord'],
                    patient_data['y_coord'],
                    c=[ecotype_colors[label] for label in patient_data['ground_truth_label']],
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                ax.set_aspect('equal')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title(f'{patient_id} - Ground Truth Spatial Map')
                ax.invert_yaxis()
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=ecotype_colors[i], edgecolor='black', label=ecotype_names_display[i])
                    for i in range(5)
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'{patient_id}_spatial_ground_truth.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    ✓ {patient_id}_spatial_ground_truth.png")
                
                # 2. Predictions Spatial Map
                fig, ax = plt.subplots(figsize=(10, 10))
                
                scatter = ax.scatter(
                    patient_data['x_coord'],
                    patient_data['y_coord'],
                    c=[ecotype_colors[label] for label in patient_data['predicted_label']],
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                ax.set_aspect('equal')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title(f'{patient_id} - Model Predictions Spatial Map')
                ax.invert_yaxis()
                
                # Add legend
                legend_elements = [
                    Patch(facecolor=ecotype_colors[i], edgecolor='black', label=ecotype_names_display[i])
                    for i in range(5)
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'{patient_id}_spatial_predictions.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    ✓ {patient_id}_spatial_predictions.png")
                
                # 3. Accuracy Map (Correct vs Incorrect predictions)
                fig, ax = plt.subplots(figsize=(10, 10))
                
                is_correct = patient_data['predicted_label'] == patient_data['ground_truth_label']
                colors = ['#2ECC71' if correct else '#E74C3C' for correct in is_correct]
                
                scatter = ax.scatter(
                    patient_data['x_coord'],
                    patient_data['y_coord'],
                    c=colors,
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                ax.set_aspect('equal')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                accuracy = is_correct.sum() / len(is_correct)
                ax.set_title(f'{patient_id} - Prediction Accuracy Map (Acc: {accuracy:.2%})')
                ax.invert_yaxis()
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ECC71', edgecolor='black', label='Correct'),
                    Patch(facecolor='#E74C3C', edgecolor='black', label='Incorrect')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'{patient_id}_spatial_accuracy.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    ✓ {patient_id}_spatial_accuracy.png")
                
                # 4. Per-Ecotype Neighborhood Analysis (simple version)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ecotype_counts = patient_data['predicted_label'].value_counts().sort_index()
                
                bars = ax.bar(
                    [ecotype_names_display[i] for i in ecotype_counts.index],
                    ecotype_counts.values,
                    color=[ecotype_colors[i] for i in ecotype_counts.index],
                    alpha=0.7,
                    edgecolor='black'
                )
                
                ax.set_ylabel('Number of Spots')
                ax.set_xlabel('Ecotype')
                ax.set_title(f'{patient_id} - Ecotype Distribution (Predicted)')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'{patient_id}_ecotype_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    ✓ {patient_id}_ecotype_distribution.png")
            
            print(f"\n  ✓ Spatial visualizations saved to: {viz_dir}")
            
        except Exception as e:
            print(f"  ⚠ Error generating spatial visualizations: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[STAGE 8] Skipping Spatial Visualizations")
        print("─" * 100)
        print("  ⚠ No spatial coordinates available. Spatial visualizations require x_coord and y_coord data.")
        print(f"  Expected metadata file: {Path(args.labels).parent / 'barcode_metadata.csv'}")
    
    print()
    
    # Summary
    print("="*100)
    print("✅ TRAINING & METRICS EXPORT COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • training_results.json           (all metrics)")
    print(f"  • 01_confusion_matrix.png         (predictions vs true)")
    print(f"  • 02_training_curves.png          (loss & accuracy per fold)")
    print(f"  • 03_per_class_accuracy.png       (accuracy per ecotype)")
    print(f"  • 04_metrics_summary.png          (overall performance)")
    print(f"  • 05_fold_comparison.png          (metrics per fold)")
    print(f"  • classification_report.txt       (detailed metrics)")
    print(f"  • model_best.pt                   (trained model)")
    print(f"  • label_encoder.pkl               (classes)")
    
    print(f"\nKey Results:")
    print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  F1 Score:  {np.mean(f1_scores_list):.4f} ± {np.std(f1_scores_list):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()
