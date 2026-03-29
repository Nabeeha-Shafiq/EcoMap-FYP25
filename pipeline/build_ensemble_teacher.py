#!/usr/bin/env python3
"""
Build Ensemble Teacher Model from All 5 Folds

Purpose:
    Instead of extracting the single best fold, creates an ensemble teacher by:
    1. Loading all 5 trained fold models
    2. Averaging their weights
    3. Freezing the ensemble model
    4. Saving as teacher_model_ENSEMBLE.pt
    
    This approach:
    - Uses all available training data
    - Reduces overfitting to single fold
    - More robust for distillation

Usage:
    python pipeline/build_ensemble_teacher.py --teacher_output "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA"
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
import sys
from typing import List
import numpy as np


class EcotypeClassifier(nn.Module):
    """Multi-layer perceptron for ecotype classification"""
    
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


def build_ensemble_teacher(teacher_output_dir):
    """
    Build ensemble by averaging all fold models.
    
    Args:
        teacher_output_dir (str): Path to teacher training output directory
    
    Returns:
        tuple: (ensemble_model, metrics_summary)
    """
    
    teacher_output = Path(teacher_output_dir)
    metrics_dir = teacher_output / "training" / "metrics"
    
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
    
    # Load training results to get architecture info
    metrics_file = metrics_dir / "training_results.json"
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("=" * 80)
    print("BUILDING ENSEMBLE TEACHER FROM ALL 5 FOLDS")
    print("=" * 80)
    
    # Get per-fold accuracy to inform weighting (optional)
    per_fold_metrics = metrics['metrics']['per_fold']
    fold_accuracies = [fold['accuracy'] for fold in per_fold_metrics]
    print(f"\nFold accuracies: {fold_accuracies}")
    print(f"Mean: {np.mean(fold_accuracies):.4f}, Std: {np.std(fold_accuracies):.4f}")
    
    # Load and average all fold models
    models_dir = teacher_output / "training" / "models"
    if not models_dir.exists():
        print(f"⚠ Models directory not found at {models_dir}")
        print(f"Checking alternate location: {metrics_dir}")
        # Try to find model_best.pt or fold models
        if (metrics_dir / "model_best.pt").exists():
            print(f"Found single best model, using as teacher")
            model_path = metrics_dir / "model_best.pt"
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Create model instance
            hyperparams = metrics['hyperparameters']
            input_dim = metrics['dataset']['n_features']
            hidden_dims = hyperparams['hidden_dims']
            num_classes = metrics['dataset']['n_classes']
            dropout = hyperparams['dropout']
            
            ensemble_model = EcotypeClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout
            )
            ensemble_model.load_state_dict(state_dict)
            ensemble_model.eval()
            
            print(f"✓ Loaded best model (single model mode)")
            return ensemble_model, {'mode': 'single_best', 'accuracy': max(fold_accuracies)}
    
    # Load fold models
    fold_models = []
    for fold_idx in range(5):
        fold_model_path = models_dir / f"fold_{fold_idx}_best_model.pth"
        if not fold_model_path.exists():
            # Try alternate naming
            fold_model_path = models_dir / f"fold_{fold_idx + 1}_best_model.pth"
        
        if fold_model_path.exists():
            print(f"✓ Found fold {fold_idx} model")
            state_dict = torch.load(fold_model_path, map_location='cpu')
            fold_models.append(state_dict)
        else:
            print(f"⚠ Fold {fold_idx} model not found")
    
    if not fold_models:
        raise FileNotFoundError(f"No fold models found in {models_dir}")
    
    print(f"\nLoaded {len(fold_models)} fold models")
    
    # Get architecture from metrics
    hyperparams = metrics['hyperparameters']
    input_dim = metrics['dataset']['n_features']
    hidden_dims = hyperparams['hidden_dims']
    num_classes = metrics['dataset']['n_classes']
    dropout = hyperparams['dropout']
    
    # Create ensemble model
    ensemble_model = EcotypeClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Average weights across all folds
    print(f"\nAveraging weights from {len(fold_models)} models...")
    ensemble_state = {}
    
    for param_name in fold_models[0].keys():
        # Stack all fold weights for this parameter
        param_stack = torch.stack([fold_state[param_name] for fold_state in fold_models])
        # Average across folds
        ensemble_state[param_name] = param_stack.mean(dim=0)
    
    ensemble_model.load_state_dict(ensemble_state)
    ensemble_model.eval()
    
    # Freeze all parameters
    for param in ensemble_model.parameters():
        param.requires_grad = False
    
    print(f"✓ Ensemble model created (averaged {len(fold_models)} folds)")
    print(f"  All parameters frozen: {sum(p.requires_grad for p in ensemble_model.parameters())} trainable params (should be 0)")
    
    # Save ensemble model
    ensemble_model_path = teacher_output / "teacher_model_ENSEMBLE.pt"
    state_dict_path = teacher_output / "teacher_state_dict_ENSEMBLE.pth"
    
    torch.save(ensemble_model.state_dict(), ensemble_model_path)
    torch.save(ensemble_model.state_dict(), state_dict_path)
    
    print(f"\n✓ Saved ensemble model: {ensemble_model_path}")
    print(f"✓ Saved state dict: {state_dict_path}")
    
    # Create summary
    summary = {
        'mode': 'ensemble',
        'n_folds': len(fold_models),
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies)),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'num_classes': num_classes,
        'model_path': str(ensemble_model_path),
        'parameters_frozen': True
    }
    
    return ensemble_model, summary


def main():
    parser = argparse.ArgumentParser(description="Build ensemble teacher from all folds")
    parser.add_argument('--teacher_output', type=str, required=True,
                       help='Path to teacher training output directory')
    
    args = parser.parse_args()
    
    try:
        ensemble_model, summary = build_ensemble_teacher(args.teacher_output)
        
        # Save summary
        summary_path = Path(args.teacher_output) / "ensemble_teacher_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_path}")
        print("\n" + "=" * 80)
        print("✅ ENSEMBLE TEACHER READY FOR STUDENT DISTILLATION")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
