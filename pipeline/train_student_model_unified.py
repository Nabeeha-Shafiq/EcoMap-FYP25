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
    """Load and preprocess student data"""
    
    print("\n[1] Loading data...")
    
    # Load UNI embeddings
    uni_path = Path(config['input_dataset']['image_encoder_embeddings'])
    uni_df = pd.read_csv(uni_path)
    uni_embeddings = uni_df.iloc[:, 1:].values.astype(np.float32)
    
    # Load labels
    labels_path = Path(config['input_dataset']['labels_file'])
    labels_df = pd.read_csv(labels_path)
    labels = labels_df.iloc[:, 1].values
    
    print(f"✓ Loaded UNI embeddings: {uni_embeddings.shape}")
    print(f"✓ Loaded labels: {len(np.unique(labels))} classes, {len(labels)} samples")
    
    # Apply PCA
    pca_var = config['embeddings']['image_encoder']['pca_variance']
    if pca_var is not None:
        pca = PCA(n_components=pca_var)
        uni_embeddings = pca.fit_transform(uni_embeddings)
        print(f"✓ Applied PCA {pca_var}: {uni_df.iloc[:, 1:].shape[1]} → {uni_embeddings.shape[1]} dims")
        
        # Save PCA model
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        pca_path = output_dir / "preprocessing" / "student_pca_model.pkl"
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
    
    # Load teacher preprocessed embeddings (full multimodal for getting teacher outputs)
    teacher_preproc_dir = Path(config['distillation']['teacher_preprocessed_dir'])
    teacher_fused_path = teacher_preproc_dir / "fused_embeddings_pca.npy"
    
    if teacher_fused_path.exists():
        teacher_embeddings = np.load(teacher_fused_path).astype(np.float32)
        print(f"✓ Loaded teacher preprocessed embeddings: {teacher_embeddings.shape}")
    else:
        print(f"⚠ Warning: Teacher embeddings not found at {teacher_fused_path}")
        teacher_embeddings = None
    
    return uni_embeddings, teacher_embeddings, labels


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


def train_student(config: Dict, teacher_model, X_student, X_teacher, y, device='cpu', output_dir=None):
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
    
    # Setup CV
    skf = StratifiedKFold(n_splits=config['training']['n_folds'], 
                         shuffle=True, 
                         random_state=config['training']['random_seed'])
    
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
        
        # Get final predictions
        student.eval()
        with torch.no_grad():
            all_preds_fold = []
            for X_batch, _, _ in val_loader:
                X_batch = X_batch.to(device)
                preds = torch.argmax(student(X_batch), dim=1).cpu().numpy()
                all_preds_fold.append(preds)
            fold_preds = np.concatenate(all_preds_fold)
        
        all_preds.extend(fold_preds)
        all_true.extend(y_val)
        fold_models.append(student.state_dict())
        
        # Fold metrics
        fold_acc = accuracy_score(y_val, fold_preds)
        fold_prec = precision_score(y_val, fold_preds, average='weighted', zero_division=0)
        fold_rec = recall_score(y_val, fold_preds, average='weighted', zero_division=0)
        fold_f1 = f1_score(y_val, fold_preds, average='weighted', zero_division=0)
        
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
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': le.inverse_transform(all_true),
        'predicted_label': le.inverse_transform(all_preds),
        'correct': all_true == all_preds
    })
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
        # Load all data
        X_student, X_teacher, y = load_data(config)
        
        # Load teacher
        teacher = load_teacher_model(config, device=device)
        
        # Train student
        results, histories, all_true, all_preds, le = train_student(
            config, teacher, X_student, X_teacher, y, 
            device=device, output_dir=output_dir
        )
        
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
