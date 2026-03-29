"""
Inference predictor module.
Loads a trained model checkpoint and runs predictions.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.inference.preprocess import preprocess_image
from ml.src.data.breed_metadata import BreedMetadataStore


class CattlePredictor:
    """
    End-to-end predictor that loads a model and provides predictions
    with breed metadata enrichment.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        class_names: Optional[list[str]] = None,
        metadata_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        img_size: int = 224,
    ):
        self.img_size = img_size
        self.device = device or torch.device('cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get class names
        if class_names:
            self.class_names = class_names
        elif 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            raise ValueError("class_names must be provided or stored in checkpoint")

        # Load model
        self.model = self._load_model(checkpoint)
        self.model.eval()

        # Load breed metadata
        self.metadata_store = None
        if metadata_path:
            try:
                self.metadata_store = BreedMetadataStore(metadata_path)
            except Exception as e:
                print(f"Warning: Could not load breed metadata: {e}")

    def _load_model(self, checkpoint: dict) -> torch.nn.Module:
        """
        Load model from checkpoint. Supports both full model and state_dict.
        """
        if 'model_state_dict' in checkpoint:
            # Need to recreate the model architecture
            model_name = checkpoint.get('model_name', 'resnet')
            num_classes = len(self.class_names)

            model = self._create_model(model_name, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Assume it's a full model
            model = checkpoint

        return model.to(self.device)

    def _create_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """Create model architecture by name."""
        if model_name == 'mlp':
            from ml.src.models.mlp import CattleMLP
            return CattleMLP(num_classes=num_classes)
        elif model_name == 'cnn':
            from ml.src.models.cnn import CattleCNN
            return CattleCNN(num_classes=num_classes)
        elif model_name in ('resnet', 'resnet50'):
            from ml.src.models.resnet import CattleResNet
            return CattleResNet(num_classes=num_classes, pretrained=False, freeze_backbone=False)
        elif model_name in ('vit', 'vit_base'):
            from ml.src.models.vit import CattleViT
            return CattleViT(num_classes=num_classes, pretrained=False, freeze_backbone=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, bytes, Image.Image],
        top_k: int = 3,
    ) -> dict:
        """
        Run prediction on a single image.

        Args:
            image: file path, URL, base64 string, bytes, or PIL Image
            top_k: number of top predictions to return

        Returns:
            dict with predicted_breed, confidence, top_k predictions, breed_info, etc.
        """
        start_time = time.time()

        # Preprocess
        tensor, pil_image = preprocess_image(image, self.img_size)
        tensor = tensor.to(self.device)

        # Inference
        outputs = self.model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze()

        # Get top-k
        top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))

        top_k_results = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            breed_name = self.class_names[idx.item()]
            top_k_results.append({
                'breed': breed_name,
                'confidence': round(prob.item(), 4),
            })

        # Best prediction
        best = top_k_results[0]
        result = {
            'predicted_breed': best['breed'],
            'confidence': best['confidence'],
            'top_k': top_k_results,
            'inference_time_ms': round((time.time() - start_time) * 1000, 2),
        }

        # Add breed metadata
        if self.metadata_store:
            breed_info = self.metadata_store.get_summary_for_prediction(best['breed'])
            if breed_info:
                result['breed_info'] = breed_info

        # Add confidence warning
        if best['confidence'] < 0.5:
            result['warning'] = (
                'Prediction confidence is low. '
                'Use a clear side-view image for better results.'
            )
        elif best['confidence'] < 0.75:
            result['warning'] = (
                'Prediction confidence is moderate. '
                'Use a clear side-view image for better results.'
            )

        return result

    def predict_batch(
        self,
        images: list[Union[str, bytes, Image.Image]],
        top_k: int = 3,
    ) -> list[dict]:
        """Run predictions on a batch of images."""
        return [self.predict(img, top_k) for img in images]
