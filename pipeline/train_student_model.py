"""
Student Model Training with Knowledge Distillation

Purpose:
    Train a morphology-only (UNI embeddings) student model using a frozen teacher
    model as guidance. Uses intermediate distillation loss combining:
    - Hard targets (true labels)
    - Soft targets (teacher probability distributions)
    - Feature matching (intermediate layer alignment)

Architecture:
    - Teacher: Multi-modal (UNI + scVI + RCTD) = 213D input
    - Student: Morphology-only (UNI) = 60D input
    - Knowledge Transfer: Student learns from teacher's decision boundaries
    
Loss Function:
    L_total = α·L_hard + β·L_soft + γ·L_features
    
    where:
      L_hard = CrossEntropyLoss(student_output, true_labels)
      L_soft = KLDivergence(student_soft, teacher_soft) [with temperature scaling]
      L_features = MSE(student_hidden, teacher_hidden)

Usage:
    python pipeline/train_student_model.py --config config/modular_Student_GEO.yaml

Expected Output:
    - ./GEO Student Models/distillation_[params]/
        ├── training/
        │   ├── models/fold_*.pth
        │   └── metrics/student_metrics.json
        ├── visualizations/
        │   ├── confusion_matrices/
        │   └── training_curves.png
        └── predictions/
            └── student_predictions.csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from datetime import datetime
import logging


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class EcotypeClassifier(nn.Module):
    """Teacher model: Multi-layer perceptron for ecotype classification"""
    
    def __init__(self, input_dim, hidden_dims=None, num_classes=5, dropout=0.3):
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


class StudentMLPClassifier(nn.Module):
    """MLP classifier for student model - MATCHES TEACHER ARCHITECTURE"""
    
    def __init__(self, input_dim, hidden_dims, n_classes, dropout_rate=0.3):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [n_classes]
        layers = []
        
        for i in range(len(dims) - 2):  # All but output layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.network = nn.Sequential(*layers)
        self.n_classes = n_classes
        
    def forward(self, x):
        return self.network(x)
    
    def get_features(self, x):
        """Get intermediate features (before output layer)"""
        # Extract all but the last layer (output layer)
        features = x
        for module in self.network[:-1]:  # All but output linear
            features = module(features)
        return features


class IntermediateDistillationLoss(nn.Module):
    """
    Intermediate distillation loss with three components:
    1. Hard target loss (classification on true labels) - WITH CLASS WEIGHTS
    2. Soft target loss (match teacher probability distribution)
    3. Feature matching loss (align intermediate representations)
    """
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, temperature=4.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
        # Use class weights if provided to handle imbalanced data
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits, teacher_logits, student_features, 
                teacher_features, ground_truth_labels):
        """
        Args:
            student_logits: [B, num_classes] raw model output
            teacher_logits: [B, num_classes] frozen teacher output (detached)
            student_features: [B, hidden_dim] intermediate layer
            teacher_features: [B, hidden_dim] intermediate layer
            ground_truth_labels: [B] true class indices
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        
        # 1. Hard target loss
        hard_loss = self.ce_loss(student_logits, ground_truth_labels)
        
        # 2. Soft target loss (with temperature scaling)
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_log_soft, teacher_soft)
        
        # 3. Feature matching loss
        # Skip if dimensions don't match (expected with different architectures)
        if student_features.shape == teacher_features.shape:
            feature_loss = self.mse_loss(student_features, teacher_features.detach())
        else:
            # Dimensions mismatch - skip feature matching
            feature_loss = torch.tensor(0.0, device=student_features.device, 
                                       dtype=student_features.dtype)
        
        # Combine with weights
        total_loss = (self.alpha * hard_loss + 
                     self.beta * soft_loss + 
                     self.gamma * feature_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'feature_loss': feature_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_embeddings(config, output_dir):
    """
    Load embeddings for student training.
    
    Returns:
        - X_student: UNI embeddings for student training (18D after PCA)
        - y: Labels
        - X_for_teacher: Full preprocessed embeddings for teacher inference (163D)
        - teacher_pca_uni: UNI component from teacher's preprocessing (10D)
    """
    
    # Load UNI embeddings
    uni_path = Path(config['input_dataset']['image_encoder_embeddings'])
    uni_embeddings = pd.read_csv(uni_path)
    
    # Load labels
    labels_path = Path(config['input_dataset']['labels_file'])
    labels_df = pd.read_csv(labels_path)
    
    # Assume first column is barcode, second is label
    uni_embeddings_array = uni_embeddings.iloc[:, 1:].values
    labels_array = labels_df.iloc[:, 1].values
    
    print(f"✓ Loaded UNI embeddings: {uni_embeddings_array.shape}")
    print(f"✓ Loaded labels: {labels_array.shape}")
    
    # Apply PCA for STUDENT input if configured
    pca_variance = config['embeddings']['image_encoder']['pca_variance']
    
    if pca_variance is not None:
        pca_student = PCA(n_components=pca_variance)
        X_student = pca_student.fit_transform(uni_embeddings_array)
        print(f"✓ Applied {pca_variance} PCA for student: {uni_embeddings_array.shape[1]} → {X_student.shape[1]} dimensions")
        
        # Save PCA model
        pca_output = output_dir / "preprocessing" / "student_pca_model.pkl"
        pca_output.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(pca_student, pca_output)
    else:
        X_student = uni_embeddings_array
    
    # Load preprocessed embeddings from teacher for knowledge distillation
    # These have already been PCA-transformed by the teacher pipeline
    teacher_output_dir = Path(config['distillation']['teacher_model_path']).parent
    teacher_fused_path = teacher_output_dir / ".working" / "preprocessed_arrays" / "fused_embeddings_pca.npy"
    
    if teacher_fused_path.exists():
        X_for_teacher = np.load(teacher_fused_path)
        print(f"✓ Loaded teacher's preprocessed fused embeddings: {X_for_teacher.shape}")
        print(f"  (This will be used for teacher inference during distillation)")
    else:
        print(f"⚠ Warning: Could not find teacher's preprocessed embeddings at {teacher_fused_path}")
        print(f"  Will fall back to multimodal loading...")
        X_for_teacher = None
    
    return X_student, labels_array, X_for_teacher


def load_teacher_model(teacher_path, device):
    """Load frozen teacher model"""
    
    teacher_path = Path(teacher_path)
    
    # Instantiate teacher model
    teacher = EcotypeClassifier(input_dim=163, hidden_dims=[256, 128, 64], 
                               num_classes=5, dropout=0.3)
    teacher = teacher.to(device)
    
    # Try to load state dict
    try:
        # If teacher_path is the frozen model, get the state dict file
        if teacher_path.name == "teacher_model_FROZEN.pt":
            state_dict_path = teacher_path.parent / "teacher_state_dict.pth"
        else:
            state_dict_path = teacher_path
        
        state_dict = torch.load(state_dict_path, map_location=device)
        teacher.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Could not load teacher state dict from {state_dict_path}: {e}")
    
    teacher.eval()
    
    # Verify frozen
    for param in teacher.parameters():
        param.requires_grad = False
        assert not param.requires_grad, "Teacher should be frozen!"
    
    print(f"✓ Loaded frozen teacher from {teacher_path}")
    return teacher


def get_teacher_outputs(teacher_model, X, device, batch_size=32):
    """
    Get teacher logits and features for entire dataset
    (cached to avoid recomputing during training)
    
    Args:
        teacher_model: Frozen teacher
        X: Input embeddings [N, D]
        device: torch device
        batch_size: Batch size for inference
        
    Returns:
        tuple: (teacher_logits, teacher_features)
    """
    
    teacher_model.eval()
    
    teacher_logits_list = []
    teacher_features_list = []
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            
            # Get logits
            logits = teacher_model(batch_X)
            teacher_logits_list.append(logits.cpu())
            
            # Get intermediate features (assuming model has method)
            # If not, just use input as features (suboptimal but functional)
            try:
                features = teacher_model.get_features(batch_X)
            except AttributeError:
                features = batch_X  # Fallback
            
            teacher_features_list.append(features.cpu())
    
    teacher_logits = torch.cat(teacher_logits_list, dim=0)
    teacher_features = torch.cat(teacher_features_list, dim=0)
    
    print(f"✓ Cached teacher outputs: logits={teacher_logits.shape}, features={teacher_features.shape}")
    
    return teacher_logits, teacher_features


def train_epoch(model, train_loader, teacher_logits, teacher_features,
                criterion, optimizer, device, fold_idx=0):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loss_dict_sum = {'hard': 0, 'soft': 0, 'feature': 0}
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # Get corresponding teacher outputs
        batch_indices = torch.arange(batch_idx * len(X), 
                                     min((batch_idx + 1) * len(X), 
                                         len(teacher_logits)))
        teacher_logits_batch = teacher_logits[batch_indices].to(device)
        teacher_features_batch = teacher_features[batch_indices].to(device)
        
        # Forward pass
        student_logits = model(X)
        
        try:
            student_features = model.get_features(X)
        except AttributeError:
            student_features = X  # Fallback
        
        # Compute loss
        loss, loss_dict = criterion(student_logits, teacher_logits_batch,
                                   student_features, teacher_features_batch, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
        
        loss_dict_sum['hard'] += loss_dict['hard_loss']
        loss_dict_sum['soft'] += loss_dict['soft_loss']
        loss_dict_sum['feature'] += loss_dict['feature_loss']
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_acc, avg_loss_dict


def validate_epoch(model, val_loader, teacher_logits, teacher_features,
                   criterion, device):
    """Validate for one epoch"""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            
            # Get teacher outputs
            batch_indices = torch.arange(batch_idx * len(X),
                                        min((batch_idx + 1) * len(X),
                                            len(teacher_logits)))
            teacher_logits_batch = teacher_logits[batch_indices].to(device)
            teacher_features_batch = teacher_features[batch_indices].to(device)
            
            # Forward pass
            student_logits = model(X)
            
            try:
                student_features = model.get_features(X)
            except AttributeError:
                student_features = X
            
            # Compute loss
            loss, _ = criterion(student_logits, teacher_logits_batch,
                               student_features, teacher_features_batch, y)
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def train_student(config, teacher_model, X_student, y, X_for_teacher, device, output_dir, logger):
    """Train student model with 5-fold cross-validation"""
    
    n_folds = config['training']['n_folds']
    cv_metrics = {
        'folds': {},
        'overall_mean_acc': 0,
        'overall_std_acc': 0
    }
    
    fold_accuracies = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                         random_state=config['training']['random_seed'])
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_student, y)):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*80}")
        
        # Split student data (UNI only for training)
        X_train, X_val = X_student[train_idx], X_student[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Split teacher data (full multimodal for getting teacher outputs)
        if X_for_teacher is not None:
            X_train_for_teacher = X_for_teacher[train_idx]
            X_val_for_teacher = X_for_teacher[val_idx]
        else:
            print("⚠ Warning: No teacher embeddings for distillation, using student embeddings (not ideal)")
            X_train_for_teacher = X_train
            X_val_for_teacher = X_val
        
        # Create datasets
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=int(config['training']['batch_size']),
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(config['training']['batch_size']),
                               shuffle=False)
        
        # Get teacher outputs for this fold's data (using FULL multimodal embeddings)
        print(f"  Getting teacher outputs for {len(train_idx)} training samples...")
        teacher_logits_train, teacher_features_train = get_teacher_outputs(
            teacher_model, X_train_for_teacher, device, batch_size=int(config['training']['batch_size'])
        )
        print(f"  Getting teacher outputs for {len(val_idx)} validation samples...")
        teacher_logits_val, teacher_features_val = get_teacher_outputs(
            teacher_model, X_val_for_teacher, device, batch_size=int(config['training']['batch_size'])
        )
        
        # Build student model
        input_dim = X_train.shape[1]
        student_model = StudentMLPClassifier(
            input_dim=input_dim,
            hidden_dims=config['training']['hidden_dims'],
            n_classes=config['dataset']['n_classes'],
            dropout_rate=float(config['training']['dropout_rate'])
        ).to(device)
        
        # Calculate class weights for handling imbalanced data
        classes, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(classes) * counts)
        class_weights = class_weights / class_weights.sum() * len(classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Setup optimization
        optimizer = optim.Adam(student_model.parameters(),
                              lr=float(config['training']['learning_rate']),
                              weight_decay=float(config['training']['weight_decay']))
        
        criterion = IntermediateDistillationLoss(
            alpha=float(config['distillation']['alpha']),
            beta=float(config['distillation']['beta']),
            gamma=float(config['distillation']['gamma']),
            temperature=float(config['distillation']['temperature']),
            class_weights=class_weights_tensor
        )
        
        # Training loop with early stopping
        best_val_acc = 0
        patience = config['training']['early_stopping_patience']
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(int(config['training']['n_epochs'])):
            train_loss, train_acc, loss_components = train_epoch(
                student_model, train_loader, teacher_logits_train, 
                teacher_features_train, criterion, optimizer, device, fold_idx
            )
            
            val_loss, val_acc = validate_epoch(
                student_model, val_loader, teacher_logits_val,
                teacher_features_val, criterion, device
            )
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = student_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save best model for this fold
        if best_model_state is not None:
            student_model.load_state_dict(best_model_state)
        
        model_output_dir = output_dir / "training" / "models"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_output_dir / f"fold_{fold_idx}_best_model.pth"
        torch.save(student_model.state_dict(), model_path)
        
        # Store metrics
        cv_metrics['folds'][str(fold_idx)] = {
            'validation_accuracy': float(best_val_acc),
            'best_epoch': len(train_losses) - patience_counter
        }
        fold_accuracies.append(best_val_acc)
        
        print(f"✓ Fold {fold_idx+1} complete: Best Val Acc = {best_val_acc:.4f}")
    
    # Overall metrics
    cv_metrics['overall_mean_acc'] = float(np.mean(fold_accuracies))
    cv_metrics['overall_std_acc'] = float(np.std(fold_accuracies))
    cv_metrics['best_fold_index'] = int(np.argmax(fold_accuracies))
    
    # Save metrics
    metrics_dir = output_dir / "training" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / "student_metrics.json", 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Mean Accuracy (5-fold CV): {cv_metrics['overall_mean_acc']:.4f} ± {cv_metrics['overall_std_acc']:.4f}")
    print(f"Best Fold: {cv_metrics['best_fold_index']} with {cv_metrics['folds'][str(cv_metrics['best_fold_index'])]['validation_accuracy']:.4f} accuracy")
    print(f"Teacher Baseline: {config['distillation']['teacher_baseline_accuracy']:.4f}")
    print(f"Performance Gap: {config['distillation']['teacher_baseline_accuracy'] - cv_metrics['overall_mean_acc']:.4f}")
    
    return cv_metrics


def main():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure all numeric config values are proper types (fix YAML parsing issues)
    config['training']['learning_rate'] = float(config['training'].get('learning_rate', 0.001))
    config['training']['batch_size'] = int(config['training'].get('batch_size', 32))
    config['training']['weight_decay'] = float(config['training'].get('weight_decay', 1e-5))
    config['training']['dropout_rate'] = float(config['training'].get('dropout_rate', 0.3))
    config['training']['n_epochs'] = int(config['training'].get('n_epochs', 200))
    config['training']['n_folds'] = int(config['training'].get('n_folds', 5))
    config['distillation']['alpha'] = float(config['distillation'].get('alpha', 0.7))
    config['distillation']['beta'] = float(config['distillation'].get('beta', 0.2))
    config['distillation']['gamma'] = float(config['distillation'].get('gamma', 0.1))
    config['distillation']['temperature'] = float(config['distillation'].get('temperature', 4.0))
    
    # Setup output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(output_dir / 'training.log')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    print("=" * 80)
    print("STUDENT MODEL TRAINING WITH KNOWLEDGE DISTILLATION")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    
    device = torch.device(args.device)
    
    # Load data
    print("\n[1] Loading embeddings...")
    X_student, y, X_for_teacher = load_embeddings(config, output_dir)
    
    # Load teacher
    print("\n[2] Loading frozen teacher...")
    teacher_model_path = Path(config['distillation']['teacher_model_path'])
    teacher_model = load_teacher_model(teacher_model_path, device)
    
    # Train student
    print("\n[3] Training student model...")
    metrics = train_student(config, teacher_model, X_student, y, X_for_teacher, device, output_dir, logger)
    
    print(f"\n✓ Student training complete! Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
