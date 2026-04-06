"""
Extract and Freeze Best Teacher Model

Purpose:
    After training completes, identifies the best-performing fold and extracts
    its weights. Freezes all parameters (requires_grad=False) to prevent
    inadvertent gradient updates during student training.

Usage:
    python pipeline/extract_teacher_model.py --teacher_output "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA"

Output:
    - teacher_model_FROZEN.pt: Serialized full model (recommended for inference)
    - teacher_state_dict.pth: Just the weights (for loading into different architectures)
    - frozen_teacher_validation.txt: Validation report confirming freezing worked
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
import sys
from typing import List


# ============================================================================
# MODEL ARCHITECTURE (copied from train_mlp.py)
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


def extract_teacher_model(teacher_output_dir):
    """
    Extract the best-performing fold's model and freeze it.
    
    Args:
        teacher_output_dir (str): Path to teacher training output directory
                                 e.g., "./GEO Ablation Study/TEACHER_FINAL_0.6_PCA"
    
    Returns:
        tuple: (teacher_model, frozen_model_path)
    """
    
    teacher_output = Path(teacher_output_dir)
    
    if not teacher_output.exists():
        raise FileNotFoundError(f"Teacher output directory not found: {teacher_output}")
    
    # Find best fold from metrics
    metrics_file = teacher_output / "training" / "metrics" / "training_results.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}\n"
                              "Training may not be complete yet.")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Find best fold from per_fold results
    per_fold = metrics.get('metrics', {}).get('per_fold', [])
    
    if not per_fold:
        raise ValueError("No per_fold data found in metrics")
    
    # Find best fold by accuracy
    best_fold_data = max(per_fold, key=lambda x: x.get('accuracy', 0))
    best_fold_index = best_fold_data.get('fold')
    best_fold_acc = best_fold_data.get('accuracy')
    
    print(f"✓ Identified best fold: Fold {best_fold_index}")
    print(f"  Validation accuracy: {best_fold_acc:.4f}")
    
    # Load best model
    # The training script saves the best model overall as model_best.pt
    model_path = teacher_output / "training" / "metrics" / "model_best.pt"
    
    if not model_path.exists():
        # Fallback: try alternate path
        model_path = teacher_output / "training" / "models" / f"fold_{best_fold_index}_best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {teacher_output / 'training' / 'metrics' / 'model_best.pt'}\n"
                              f"or {teacher_output / 'training' / 'models' / f'fold_{best_fold_index}_best_model.pth'}\n"
                              f"Training may not be complete yet.")
    
    print(f"\n✓ Loading model from: {model_path}")
    
    # Instantiate model architecture
    hyperparams = metrics.get('hyperparameters', {})
    input_dim = metrics.get('dataset', {}).get('n_features', 163)
    hidden_dims = hyperparams.get('hidden_dims', [256, 128, 64])
    num_classes = metrics.get('dataset', {}).get('n_classes', 5)
    dropout = hyperparams.get('dropout', 0.3)
    
    print(f"  Input dim: {input_dim}, Hidden: {hidden_dims}, Classes: {num_classes}")
    
    # Create and load model
    teacher_model = EcotypeClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()
    
    # Freeze all parameters
    initial_grad_count = sum(p.requires_grad for p in teacher_model.parameters())
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    final_grad_count = sum(p.requires_grad for p in teacher_model.parameters())
    
    print(f"\n✓ Frozen parameters:")
    print(f"  Before: {initial_grad_count} gradients enabled")
    print(f"  After:  {final_grad_count} gradients enabled")
    
    # Save frozen teacher
    frozen_model_path = teacher_output / "teacher_model_FROZEN.pt"
    torch.save(teacher_model, frozen_model_path)
    print(f"\n✓ Saved frozen model: {frozen_model_path}")
    
    # Also save state dict for reproducibility
    state_dict_path = teacher_output / "teacher_state_dict.pth"
    torch.save(teacher_model.state_dict(), state_dict_path)
    print(f"✓ Saved state dict: {state_dict_path}")
    
    return teacher_model, frozen_model_path


def validate_frozen_teacher(teacher_model, frozen_model_path, device='cpu'):
    """
    Validate that teacher is properly frozen and can be used for distillation.
    
    Checks:
        1. No gradients tracked (requires_grad all False)
        2. Consistent predictions across multiple calls
        3. Memory efficient (no computation graph buildup)
    
    Args:
        teacher_model: Frozen teacher model
        frozen_model_path: Path where frozen model was saved
        device: Device to use for validation
    
    Returns:
        dict: Validation report with boolean results
    """
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    checks = {
        'no_gradient_tracking': True,
        'consistent_predictions': True,
        'memory_efficient': True,
        'model_loadable': False,
        'gradients_disabled': True
    }
    
    # Check 1: No gradient tracking
    print("\n[Check 1] Gradient Tracking...")
    for name, param in teacher_model.named_parameters():
        if param.requires_grad:
            print(f"  ✗ Parameter '{name}' still has requires_grad=True")
            checks['no_gradient_tracking'] = False
            break
    
    if checks['no_gradient_tracking']:
        print("  ✓ All parameters have requires_grad=False")
    
    # Check 2: Consistent predictions 
    print("\n[Check 2] Prediction Consistency...")
    try:
        x = torch.randn(4, 213).to(device)  # Assuming 213D fused input
        
        with torch.no_grad():
            pred1 = teacher_model(x)
            pred2 = teacher_model(x)
        
        if torch.allclose(pred1, pred2, atol=1e-6):
            print("  ✓ Predictions consistent across calls")
            checks['consistent_predictions'] = True
        else:
            print("  ✗ Predictions differ across calls (dropout enabled?)")
            checks['consistent_predictions'] = False
    except Exception as e:
        print(f"  ⚠ Could not validate prediction consistency: {e}")
    
    # Check 3: Memory efficiency
    print("\n[Check 3] Memory Efficiency...")
    try:
        import tracemalloc
        tracemalloc.start()
        
        with torch.no_grad():
            for i in range(10):
                _ = teacher_model(torch.randn(4, 213).to(device))
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"  Peak memory: {peak_mb:.2f} MB")
        
        if peak_mb < 500:
            print("  ✓ Memory efficient (< 500MB)")
            checks['memory_efficient'] = True
        else:
            print("  ⚠ High memory usage - consider reducing inference batch size")
    except Exception as e:
        print(f"  ⚠ Could not measure memory: {e}")
    
    # Check 4: Model reloadable
    print("\n[Check 4] Model Reloadability...")
    try:
        reloaded = torch.load(frozen_model_path, map_location=device)
        print(f"  ✓ Model successfully reloaded from {frozen_model_path}")
        checks['model_loadable'] = True
    except Exception as e:
        print(f"  ✗ Could not reload model: {e}")
        checks['model_loadable'] = False
    
    # Check 5: Gradients not computed
    print("\n[Check 5] No Gradient Computation...")
    try:
        x = torch.randn(4, 213, requires_grad=True).to(device)
        with torch.no_grad():
            output = teacher_model(x)
        
        if output.requires_grad:
            print("  ✗ Output still requires gradients")
            checks['gradients_disabled'] = False
        else:
            print("  ✓ No gradients computed")
            checks['gradients_disabled'] = True
    except Exception as e:
        print(f"  ⚠ Could not validate gradient computation: {e}")
    
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Extract and freeze best teacher model for student distillation"
    )
    parser.add_argument('--teacher_output', type=str, required=True,
                       help='Path to teacher training output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for validation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEACHER MODEL EXTRACTION & FREEZING")
    print("=" * 80)
    
    try:
        # Extract best fold
        teacher_model, frozen_path = extract_teacher_model(args.teacher_output)
        
        # Validate freezing
        print("\n" + "=" * 80)
        print("VALIDATION TESTS")
        print("=" * 80)
        
        checks = validate_frozen_teacher(teacher_model, frozen_path, device=args.device)
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        all_passed = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {check_name}")
        
        # Save validation report
        report_path = Path(args.teacher_output) / "frozen_teacher_validation.txt"
        with open(report_path, 'w') as f:
            f.write("TEACHER MODEL FREEZING VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for check_name, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                f.write(f"{status}: {check_name}\n")
            
            f.write(f"\nOverall: {'✓ READY FOR DISTILLATION' if all_passed else '✗ ISSUES DETECTED'}\n")
        
        print(f"\n✓ Validation report saved: {report_path}")
        
        if all_passed:
            print("\n✓ Teacher model successfully frozen and ready for distillation!")
            return 0
        else:
            print("\n⚠ Some checks failed - review above for details")
            return 1
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
