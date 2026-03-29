"""
Loss function factory.
Supports CrossEntropy with optional label smoothing and focal loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_loss(
    loss_type: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: 'cross_entropy' or 'focal'
        label_smoothing: smoothing factor (0 = none)
        focal_alpha: alpha for focal loss
        focal_gamma: gamma for focal loss
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
