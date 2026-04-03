"""
ResNet transfer learning model.
Wraps torchvision ResNet50 with custom classifier head.
Supports two-phase training: frozen backbone → partial fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class CattleResNet(nn.Module):
    """
    ResNet-based classifier with transfer learning support.
    Phase 1: Freeze all backbone layers, train only the classifier head.
    Phase 2: Unfreeze specified layers for fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 26,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone

        # Load pretrained backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_layers(self, layer_names: list[str] = None) -> None:
        """
        Unfreeze specified layers for fine-tuning.
        Default: unfreeze layer3, layer4, and fc.
        """
        if layer_names is None:
            layer_names = ['layer3', 'layer4', 'fc']

        for name, param in self.model.named_parameters():
            if any(ln in name for ln in layer_names):
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list:
        """Return list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_param_groups(self, base_lr: float, fine_tune_lr: float) -> list[dict]:
        """
        Return parameter groups with different learning rates.
        Backbone gets fine_tune_lr, head gets base_lr.
        """
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': fine_tune_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr})

        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def from_config(config: dict) -> 'CattleResNet':
        """Create model from config dict."""
        arch = config.get('model', {}).get('architecture', {})
        return CattleResNet(
            num_classes=config.get('num_classes', 26),
            backbone=arch.get('backbone', 'resnet50'),
            pretrained=arch.get('pretrained', True),
            freeze_backbone=arch.get('freeze_backbone', True),
        )
