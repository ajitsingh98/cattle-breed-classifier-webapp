"""
Unified Trainer class that orchestrates the full training pipeline.
Takes model, dataloaders, config → trains, evaluates, saves artifacts.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.training.engine import train_one_epoch, validate, measure_inference_latency
from ml.src.training.losses import create_loss
from ml.src.training.optimizer_factory import create_optimizer
from ml.src.training.scheduler_factory import create_scheduler
from ml.src.training.callbacks import EarlyStopping, ModelCheckpoint, MetricLogger
from ml.src.evaluation.metrics import compute_metrics
from ml.src.evaluation.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_f1,
)
from ml.src.utils.io import save_json, ensure_dirs
from ml.src.utils.seed import set_seed


class Trainer:
    """
    Unified training orchestrator.

    Usage:
        trainer = Trainer(model, config, dataloaders, class_names, device)
        trainer.train()
        trainer.evaluate()
        trainer.save_artifacts()
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        dataloaders: dict,
        class_names: list[str],
        device: torch.device,
        model_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.dataloaders = dataloaders
        self.class_names = class_names
        self.device = device
        self.model_name = model_name or config.get('model', {}).get('name', 'model')

        # Training config
        train_cfg = config.get('training', {})
        self.num_epochs = train_cfg.get('num_epochs', 30)
        self.gradient_clip = train_cfg.get('gradient_clip_max_norm', 1.0)

        # Setup artifact directories
        artifacts_cfg = config.get('artifacts', {})
        self.checkpoint_dir = PROJECT_ROOT / artifacts_cfg.get('checkpoints_dir', 'ml/artifacts/checkpoints')
        self.figures_dir = PROJECT_ROOT / artifacts_cfg.get('figures_dir', 'ml/artifacts/figures')
        self.logs_dir = PROJECT_ROOT / artifacts_cfg.get('logs_dir', 'ml/artifacts/logs')
        self.reports_dir = PROJECT_ROOT / artifacts_cfg.get('reports_dir', 'ml/artifacts/reports')
        ensure_dirs(self.checkpoint_dir, self.figures_dir, self.logs_dir, self.reports_dir)

        # Create loss
        label_smoothing = train_cfg.get('label_smoothing', 0.0)
        self.criterion = create_loss(label_smoothing=label_smoothing)

        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=train_cfg.get('optimizer', 'adam'),
            learning_rate=train_cfg.get('learning_rate', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )

        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_name=train_cfg.get('scheduler', 'cosine'),
            num_epochs=self.num_epochs,
            step_size=train_cfg.get('scheduler_step_size', 10),
            gamma=train_cfg.get('scheduler_gamma', 0.1),
            warmup_epochs=train_cfg.get('warmup_epochs', 0),
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get('early_stopping_patience', 7),
            mode='min',
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=self.checkpoint_dir,
            model_name=self.model_name,
            mode='min',
        )
        self.logger = MetricLogger(
            log_dir=self.logs_dir,
            model_name=self.model_name,
        )

        # Results storage
        self.test_results = None
        self.training_summary = None

    def train(self) -> dict:
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

        set_seed(self.config.get('data', {}).get('seed', 42))

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = train_one_epoch(
                self.model, self.dataloaders['train'], self.criterion,
                self.optimizer, self.device, self.gradient_clip,
            )

            # Validate
            val_metrics = validate(
                self.model, self.dataloaders['val'], self.criterion, self.device,
            )

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            self.logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Checkpoint
            self.checkpoint(
                val_metrics['loss'], self.model, self.optimizer, epoch,
                extra_info={'config': self.config, 'class_names': self.class_names},
            )

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save training history
        self.logger.save()
        self.training_summary = self.logger.get_summary()

        print(f"\nTraining complete! Best val loss: {self.checkpoint.best_value:.4f}")
        return self.training_summary

    def evaluate(self, split: str = 'test') -> dict:
        """
        Evaluate the best model on test set.
        Loads the best checkpoint and runs full evaluation.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {self.model_name} on {split}")
        print(f"{'='*60}\n")

        # Load best checkpoint
        if self.checkpoint.best_path and self.checkpoint.best_path.exists():
            ckpt = torch.load(self.checkpoint.best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

        # Run evaluation
        results = validate(
            self.model, self.dataloaders[split], self.criterion, self.device,
        )

        # Compute detailed metrics
        metrics = compute_metrics(
            results['labels'], results['predictions'], self.class_names,
        )

        # Measure inference latency
        latency = measure_inference_latency(
            self.model, self.device,
            img_size=self.config.get('image', {}).get('size', 224),
        )

        # Calculate model size
        model_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        self.test_results = {
            'model_name': self.model_name,
            'split': split,
            'loss': results['loss'],
            'metrics': metrics,
            'latency': latency,
            'model_size_mb': round(model_size_mb, 2),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }

        # Print summary
        print(f"\n{split.upper()} Results:")
        print(f"  Loss:          {results['loss']:.4f}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
        print(f"  Macro Prec:    {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:  {metrics['macro_recall']:.4f}")
        print(f"  Avg Latency:   {latency['avg_ms']:.2f} ms")
        print(f"  Model Size:    {model_size_mb:.2f} MB")

        return self.test_results

    def save_artifacts(self) -> None:
        """Save all evaluation artifacts: plots, reports."""
        print(f"\nSaving artifacts for {self.model_name}...")

        model_fig_dir = self.figures_dir / self.model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)

        # Training curves
        plot_training_curves(
            self.logger.history,
            save_path=model_fig_dir / 'training_curves.png',
            title=f'{self.model_name} - Training Curves',
        )

        # Confusion matrix (from test results)
        if self.test_results:
            labels = self.test_results.get('_raw_labels', None)
            preds = self.test_results.get('_raw_predictions', None)

            # Load from last validation run if needed
            if labels is None:
                val_results = validate(
                    self.model, self.dataloaders.get('test', self.dataloaders['val']),
                    self.criterion, self.device,
                )
                labels = val_results['labels']
                preds = val_results['predictions']

            plot_confusion_matrix(
                labels, preds, self.class_names,
                save_path=model_fig_dir / 'confusion_matrix.png',
                title=f'{self.model_name} - Confusion Matrix',
            )

            # Per-class F1
            if self.test_results and 'metrics' in self.test_results:
                per_class = self.test_results['metrics'].get('per_class', {})
                if per_class:
                    plot_per_class_f1(
                        per_class, self.class_names,
                        save_path=model_fig_dir / 'per_class_f1.png',
                        title=f'{self.model_name} - Per-Class F1 Score',
                    )

        # Save detailed report JSON
        if self.test_results:
            report = {**self.test_results}
            report.pop('_raw_labels', None)
            report.pop('_raw_predictions', None)
            save_json(report, self.reports_dir / f'{self.model_name}_report.json')

        # Save training summary
        if self.training_summary:
            save_json(
                self.training_summary,
                self.reports_dir / f'{self.model_name}_training_summary.json',
            )

        print(f"  Artifacts saved to {self.figures_dir / self.model_name}")

    def train_with_phase_switch(
        self,
        phase1_epochs: int = 10,
        phase2_lr: float = 0.0001,
        unfreeze_layers: list[str] = None,
    ) -> dict:
        """
        Two-phase training for transfer learning models.
        Phase 1: Frozen backbone, train head only.
        Phase 2: Unfreeze specified layers, use lower LR.
        """
        print(f"\n{'='*60}")
        print(f"Phase 1: Frozen backbone training ({phase1_epochs} epochs)")
        print(f"{'='*60}\n")

        # Phase 1
        original_epochs = self.num_epochs
        self.num_epochs = phase1_epochs
        self.train()

        # Phase 2: Unfreeze
        remaining_epochs = original_epochs - phase1_epochs
        if remaining_epochs > 0 and hasattr(self.model, 'unfreeze_layers'):
            print(f"\n{'='*60}")
            print(f"Phase 2: Fine-tuning ({remaining_epochs} epochs)")
            print(f"{'='*60}\n")

            self.model.unfreeze_layers(unfreeze_layers)

            # Use param groups if available
            if hasattr(self.model, 'get_param_groups'):
                base_lr = self.config.get('training', {}).get('learning_rate', 0.001)
                param_groups = self.model.get_param_groups(base_lr, phase2_lr)
                self.optimizer = create_optimizer(
                    param_groups,
                    optimizer_name=self.config.get('training', {}).get('optimizer', 'adamw'),
                    learning_rate=base_lr,
                    weight_decay=self.config.get('training', {}).get('weight_decay', 0.0001),
                )
            else:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = phase2_lr

            # Reset scheduler and early stopping
            self.scheduler = create_scheduler(
                self.optimizer,
                scheduler_name=self.config.get('training', {}).get('scheduler', 'cosine'),
                num_epochs=remaining_epochs,
            )
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training', {}).get('early_stopping_patience', 7),
            )

            self.num_epochs = remaining_epochs
            self.train()

        self.num_epochs = original_epochs
        return self.logger.get_summary()
