"""
Custom CNN model built from scratch.
5 convolutional blocks with batch norm + global average pooling.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CattleCNN(nn.Module):
    """
    Custom CNN with 5 convolutional blocks.
    Each block: 2x(Conv2d + BN + ReLU) + MaxPool
    Ends with Global Average Pooling + FC classifier.
    """

    def __init__(
        self,
        num_classes: int = 26,
        in_channels: int = 3,
        conv_channels: list[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        dropout: float = 0.4,
        use_global_avg_pool: bool = True,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128, 256, 512]

        self.use_global_avg_pool = use_global_avg_pool

        # Build conv blocks
        blocks = []
        prev_channels = in_channels
        for out_ch in conv_channels:
            blocks.append(ConvBlock(prev_channels, out_ch, kernel_size, padding, pool_size))
            prev_channels = out_ch

        self.features = nn.Sequential(*blocks)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def from_config(config: dict) -> 'CattleCNN':
        """Create model from config dict."""
        arch = config.get('model', {}).get('architecture', {})
        return CattleCNN(
            num_classes=config.get('num_classes', 26),
            in_channels=config.get('image', {}).get('channels', 3),
            conv_channels=arch.get('conv_channels', [32, 64, 128, 256, 512]),
            kernel_size=arch.get('kernel_size', 3),
            padding=arch.get('padding', 1),
            pool_size=arch.get('pool_size', 2),
            dropout=arch.get('dropout', 0.4),
            use_global_avg_pool=arch.get('use_global_avg_pool', True),
        )
