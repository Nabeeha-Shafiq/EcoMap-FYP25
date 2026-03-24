#!/usr/bin/env python3
"""
Training Metrics Tracker
========================

Purpose:
  Track and log training metrics during MLP training with 5-fold cross-validation

Features:
  - Per-fold training/validation loss tracking
  - Early stopping detection and logging
  - Fold-level accuracy metrics
  - Per-class metrics calculation
  - CSV export for reproducibility

Usage:
  tracker = MetricsTracker(save_dir='results/training')
  # In training loop:
  tracker.log_epoch(fold, epoch, train_loss, val_loss, is_best=True)
  tracker.log_fold_complete(fold, accuracy, precision, recall, f1)
  # After training:
  tracker.save_metrics_csv()
  tracker.save_fold_results()

Author: Metrics Pipeline
Date: March 24, 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json
from datetime import datetime


@dataclass
class EpochMetrics:
    """Single epoch metrics."""
    fold: int
    epoch: int
    train_loss: float
    val_loss: float
    is_best: bool = False
    early_stop: bool = False


@dataclass
class FoldMetrics:
    """Metrics for one fold after training complete."""
    fold: int
    final_epoch: int
    best_epoch: int
    train_loss_final: float
    val_loss_final: float
    val_loss_best: float
    early_stopped: bool
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)


class MetricsTracker:
    """
    Track metrics during training for reproducible ablation studies.
    """
    
    def __init__(self, save_dir: str = 'results/training'):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.epoch_metrics: List[EpochMetrics] = []
        self.fold_metrics: List[FoldMetrics] = []
        self.fold_histories: Dict[int, Dict] = {}  # Per-fold training history
        
        # Current fold tracking
        self.current_fold = None
        self.current_fold_data = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'patience_counter': 0
        }
        
        # Metadata
        self.start_time = datetime.now()
        self.config = {}
        
        print(f"✓ MetricsTracker initialized")
        print(f"  Output directory: {self.save_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EPOCH LOGGING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float, 
                  is_best: bool = False, early_stop: bool = False):
        """
        Log metrics for a single epoch.
        
        Args:
            fold: Fold number (0-4)
            epoch: Epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            is_best: Whether this is the best epoch so far
            early_stop: Whether early stopping triggered
        """
        # Track epoch
        metric = EpochMetrics(
            fold=fold,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=is_best,
            early_stop=early_stop
        )
        self.epoch_metrics.append(metric)
        
        # Update fold history
        if fold != self.current_fold:
            if self.current_fold is not None:
                # Save previous fold
                self.fold_histories[self.current_fold] = self.current_fold_data.copy()
            
            self.current_fold = fold
            self.current_fold_data = {
                'epoch': [],
                'train_loss': [],
                'val_loss': [],
                'best_epoch': epoch if is_best else 0,
                'best_val_loss': val_loss if is_best else float('inf'),
                'patience_counter': 0
            }
        
        # Update current fold data
        self.current_fold_data['epoch'].append(epoch)
        self.current_fold_data['train_loss'].append(train_loss)
        self.current_fold_data['val_loss'].append(val_loss)
        
        if is_best:
            self.current_fold_data['best_epoch'] = epoch
            self.current_fold_data['best_val_loss'] = val_loss
        
        if early_stop:
            self.current_fold_data['early_stopped'] = True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOLD COMPLETION LOGGING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_fold_complete(self, fold: int, accuracy: float, precision: float, 
                         recall: float, f1_score: float, 
                         per_class_accuracy: Dict[int, float] = None):
        """
        Log metrics when fold training is complete.
        
        Args:
            fold: Fold number
            accuracy: Overall accuracy
            precision: Overall precision
            recall: Overall recall
            f1_score: Overall F1 score
            per_class_accuracy: Dict mapping class_id → accuracy
        """
        # Get last epoch metrics for this fold
        fold_epochs = [m for m in self.epoch_metrics if m.fold == fold]
        
        if not fold_epochs:
            print(f"⚠ Warning: No epoch metrics found for fold {fold}")
            return
        
        last_epoch = fold_epochs[-1]
        best_epochs = [m for m in fold_epochs if m.is_best]
        
        if best_epochs:
            best_epoch = best_epochs[-1].epoch
            best_val_loss = best_epochs[-1].val_loss
        else:
            best_epoch = last_epoch.epoch
            best_val_loss = last_epoch.val_loss
        
        # Save fold history
        if self.current_fold == fold:
            self.fold_histories[fold] = self.current_fold_data.copy()
        
        # Create fold metrics
        fold_metric = FoldMetrics(
            fold=fold,
            final_epoch=last_epoch.epoch,
            best_epoch=best_epoch,
            train_loss_final=last_epoch.train_loss,
            val_loss_final=last_epoch.val_loss,
            val_loss_best=best_val_loss,
            early_stopped=any(m.early_stop for m in fold_epochs),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            per_class_accuracy=per_class_accuracy or {}
        )
        
        self.fold_metrics.append(fold_metric)
        
        print(f"✓ Fold {fold} complete:")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
        print(f"  Best epoch: {best_epoch}, Early stopped: {fold_metric.early_stopped}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVING & EXPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_metrics_csv(self, filename: str = 'training_metrics.csv'):
        """
        Save all epoch metrics to CSV.
        
        Output CSV has columns:
            fold, epoch, train_loss, val_loss, is_best, early_stop
        """
        if not self.epoch_metrics:
            print("⚠ No epoch metrics to save")
            return
        
        # Convert to DataFrame
        data = [asdict(m) for m in self.epoch_metrics]
        df = pd.DataFrame(data)
        
        # Save
        output_file = self.save_dir / filename
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved training metrics: {output_file.name}")
        print(f"  Total epochs: {len(df)}")
    
    def save_fold_results(self, filename: str = 'fold_results.csv'):
        """
        Save per-fold final results to CSV.
        
        Output CSV has columns:
            fold, final_epoch, best_epoch, accuracy, precision, recall, f1_score, 
            early_stopped, val_loss_best
        """
        if not self.fold_metrics:
            print("⚠ No fold metrics to save")
            return
        
        # Convert to DataFrame
        data = []
        for fm in self.fold_metrics:
            row = {
                'fold': fm.fold,
                'final_epoch': fm.final_epoch,
                'best_epoch': fm.best_epoch,
                'train_loss_final': fm.train_loss_final,
                'val_loss_final': fm.val_loss_final,
                'val_loss_best': fm.val_loss_best,
                'early_stopped': fm.early_stopped,
                'accuracy': fm.accuracy,
                'precision': fm.precision,
                'recall': fm.recall,
                'f1_score': fm.f1_score
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save
        output_file = self.save_dir / filename
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved fold results: {output_file.name}")
        
        # Print summary statistics
        self._print_summary_stats(df)
    
    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics from fold results."""
        print(f"\n📊 Cross-Validation Summary:")
        print(f"  Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        print(f"  Recall: {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
        print(f"  F1 Score: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
        print(f"  Early Stops: {df['early_stopped'].sum()}/{len(df)}")
    
    def save_fold_histories(self, filename: str = 'fold_histories.json'):
        """
        Save per-fold training histories to JSON.
        
        Useful for plotting training curves.
        """
        if not self.fold_histories:
            print("⚠ No fold histories to save")
            return
        
        # Make JSON serializable
        histories = {}
        for fold, history in self.fold_histories.items():
            histories[str(fold)] = {
                'epoch': [int(e) for e in history.get('epoch', [])],
                'train_loss': [float(l) for l in history.get('train_loss', [])],
                'val_loss': [float(l) for l in history.get('val_loss', [])],
                'best_epoch': int(history.get('best_epoch', 0)),
                'best_val_loss': float(history.get('best_val_loss', float('inf'))),
                'early_stopped': history.get('early_stopped', False)
            }
        
        output_file = self.save_dir / filename
        with open(output_file, 'w') as f:
            json.dump(histories, f, indent=2)
        
        print(f"✓ Saved fold histories: {output_file.name}")
    
    def save_summary(self, filename: str = 'training_summary.json'):
        """
        Save overall training summary.
        """
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_folds': int(len(self.fold_metrics)),
            'total_epochs_tracked': int(len(self.epoch_metrics)),
            'config': self._convert_to_serializable(self.config.copy() if self.config else {})
        }
        
        # Add fold statistics
        if self.fold_metrics:
            fold_df = pd.DataFrame([asdict(fm) for fm in self.fold_metrics])
            summary['fold_stats'] = {
                'accuracy_mean': float(fold_df['accuracy'].mean()),
                'accuracy_std': float(fold_df['accuracy'].std()),
                'f1_mean': float(fold_df['f1_score'].mean()),
                'f1_std': float(fold_df['f1_score'].std()),
                'early_stops': int(fold_df['early_stopped'].sum())
            }
        
        output_file = self.save_dir / filename
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved training summary: {output_file.name}")
    
    def export_all(self):
        """Export all metrics and histories."""
        print("\n" + "="*80)
        print("EXPORTING ALL TRAINING METRICS")
        print("="*80)
        
        self.save_metrics_csv()
        self.save_fold_results()
        self.save_fold_histories()
        self.save_summary()
        
        print("\n✓ All metrics exported successfully")
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GETTERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_fold_metrics(self, fold: int) -> FoldMetrics:
        """Get metrics for a specific fold."""
        for fm in self.fold_metrics:
            if fm.fold == fold:
                return fm
        return None
    
    def get_fold_history(self, fold: int) -> Dict:
        """Get training history for a specific fold."""
        return self.fold_histories.get(fold, {})
    
    def get_cross_validation_summary(self) -> pd.DataFrame:
        """Get cross-validation summary as DataFrame."""
        if not self.fold_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame([asdict(fm) for fm in self.fold_metrics])
