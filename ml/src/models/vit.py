"""
Vision Transformer (ViT) transfer learning model.
Uses timm library for pretrained ViT-B/16.
Supports selective layer unfreezing and differential LR.
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. ViT model requires: pip install timm")


class CattleViT(nn.Module):
    """
    Vision Transformer classifier using pretrained ViT from timm.
    Phase 1: Freeze all blocks, train only the classification head.
    Phase 2: Selectively unfreeze last N blocks for fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 26,
        backbone: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT. Install with: pip install timm")

        self.backbone_name = backbone

        # Load pretrained ViT
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        num_features = self.model.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, layer_names: list[str] = None) -> None:
        """
        Unfreeze specified layers for fine-tuning.
        Default: last 2 transformer blocks + norm layer.
        """
        if layer_names is None:
            layer_names = ['blocks.10', 'blocks.11', 'norm']

        for name, param in self.model.named_parameters():
            if any(ln in name for ln in layer_names):
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list:
        """Return list of all trainable parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        params += [p for p in self.head.parameters() if p.requires_grad]
        return params

    def get_param_groups(self, base_lr: float, fine_tune_lr: float) -> list[dict]:
        """
        Return parameter groups with different learning rates.
        Backbone gets fine_tune_lr, head gets base_lr.
        """
        backbone_params = [p for p in self.model.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': fine_tune_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr})

        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return self.head(features)

    @staticmethod
    def from_config(config: dict) -> 'CattleViT':
        """Create model from config dict."""
        arch = config.get('model', {}).get('architecture', {})
        return CattleViT(
            num_classes=config.get('num_classes', 26),
            backbone=arch.get('backbone', 'vit_base_patch16_224'),
            pretrained=arch.get('pretrained', True),
            freeze_backbone=arch.get('freeze_backbone', True),
        )
