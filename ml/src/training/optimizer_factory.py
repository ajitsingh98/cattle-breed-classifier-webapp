"""
Optimizer factory. Creates optimizers from config strings.
"""

import torch.optim as optim
import torch.nn as nn


def create_optimizer(
    model_or_params,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Factory function to create optimizer.

    Args:
        model_or_params: nn.Module or list of parameters/param groups
        optimizer_name: 'adam', 'adamw', 'sgd', or 'rmsprop'
        learning_rate: base learning rate
        weight_decay: L2 regularization
        momentum: momentum for SGD/RMSProp
    """
    # Get parameters
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()
    elif isinstance(model_or_params, list) and len(model_or_params) > 0:
        if isinstance(model_or_params[0], dict):
            # Already param groups
            params = model_or_params
        else:
            params = model_or_params
    else:
        params = model_or_params

    name = optimizer_name.lower()

    if name == 'adam':
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay,
                         momentum=momentum, nesterov=True)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay,
                             momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Use 'adam', 'adamw', 'sgd', or 'rmsprop'.")
