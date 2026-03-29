"""
Learning rate scheduler factory.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 30,
    step_size: int = 10,
    gamma: float = 0.1,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> optim.lr_scheduler.LRScheduler | None:
    """
    Factory function to create learning rate scheduler.

    Args:
        optimizer: optimizer instance
        scheduler_name: 'cosine', 'step', 'plateau', 'cosine_restart', or 'none'
        num_epochs: total number of training epochs
        step_size: step size for StepLR
        gamma: decay factor for StepLR
        warmup_epochs: number of warmup epochs (linear warmup)
        min_lr: minimum learning rate for cosine schedulers
    """
    name = scheduler_name.lower()

    if name == 'none':
        return None

    # Create the main scheduler
    if name == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr,
        )
    elif name == 'step':
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'plateau':
        main_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=gamma, patience=5, min_lr=min_lr,
        )
    elif name == 'cosine_restart':
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    # Add warmup if requested
    if warmup_epochs > 0 and name != 'plateau':
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        return scheduler

    return main_scheduler
