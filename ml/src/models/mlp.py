"""
MLP (Multi-Layer Perceptron) baseline model.
Flattens input image and passes through dense layers.
Purpose: weak baseline to compare against CNNs and transformers.
"""

import torch
import torch.nn as nn


class CattleMLP(nn.Module):
    """
    Simple MLP classifier for image classification.
    Flattens the image into a vector and passes through FC layers.
    """

    def __init__(
        self,
        num_classes: int = 26,
        img_size: int = 224,
        channels: int = 3,
        hidden_layers: list[int] = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [1024, 512, 256]

        self.flatten = nn.Flatten()
        input_dim = channels * img_size * img_size

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Final classifier
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def from_config(config: dict) -> 'CattleMLP':
        """Create model from config dict."""
        arch = config.get('model', {}).get('architecture', {})
        return CattleMLP(
            num_classes=config.get('num_classes', 26),
            img_size=config.get('image', {}).get('size', 224),
            channels=config.get('image', {}).get('channels', 3),
            hidden_layers=arch.get('hidden_layers', [1024, 512, 256]),
            dropout=arch.get('dropout', 0.3),
            batch_norm=arch.get('batch_norm', True),
        )
