"""
Training callbacks: EarlyStopping, ModelCheckpoint, MetricLogger.
"""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  EarlyStopping triggered!")
                return True

        return False


class ModelCheckpoint:
    """Save model checkpoint when monitored metric improves."""

    def __init__(
        self,
        save_dir: str | Path,
        model_name: str = 'model',
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.verbose = verbose
        self.best_value = None
        self.best_path = None

    def __call__(
        self,
        value: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = 0,
        extra_info: Optional[dict] = None,
    ) -> bool:
        if self.best_value is None:
            improved = True
        elif self.mode == 'min':
            improved = value < self.best_value
        else:
            improved = value > self.best_value

        if improved:
            self.best_value = value
            filename = f"{self.model_name}_best.pth"
            self.best_path = self.save_dir / filename

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_value': self.best_value,
                'model_name': self.model_name,
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if extra_info:
                checkpoint.update(extra_info)

            torch.save(checkpoint, self.best_path)

            if self.verbose:
                print(f"  Checkpoint saved: {filename} (value: {value:.4f})")

            return True

        return False


class MetricLogger:
    """Log and store metrics across epochs."""

    def __init__(self, log_dir: str | Path = None, model_name: str = 'model'):
        self.model_name = model_name
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        self.start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        learning_rate: float,
    ) -> None:
        """Log metrics for one epoch."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['learning_rate'].append(learning_rate)
        self.history['epoch_time'].append(train_metrics.get('time_seconds', 0))

        print(f"  Epoch {epoch:>3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {learning_rate:.6f}")

    def save(self) -> None:
        """Save training history to JSON."""
        if self.log_dir:
            total_time = time.time() - self.start_time
            summary = {
                'model_name': self.model_name,
                'total_training_time_seconds': round(total_time, 2),
                'num_epochs': len(self.history['train_loss']),
                'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
                'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else None,
                'history': self.history,
            }

            path = self.log_dir / f'{self.model_name}_training_history.json'
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  Training history saved to {path}")

    def get_summary(self) -> dict:
        """Get training summary."""
        return {
            'model_name': self.model_name,
            'total_epochs': len(self.history['train_loss']),
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
            'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else None,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
        }
