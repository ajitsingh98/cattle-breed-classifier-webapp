"""
Manifest CSV dataset loader.
Loads train/val/test splits from manifest CSVs and provides PyTorch datasets.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ManifestDataset(Dataset):
    """
    PyTorch Dataset that reads from a manifest CSV.
    Each row has at minimum: image_path, class_name.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        df = pd.read_csv(manifest_path)
        self.image_paths = df['image_path'].tolist()
        self.class_names = df['class_name'].tolist()

        # Build class mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(set(self.class_names))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.labels = [self.class_to_idx[c] for c in self.class_names]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.data_root / self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def classes(self) -> list[str]:
        return [self.idx_to_class[i] for i in range(self.num_classes)]


def create_dataloaders(
    manifests_dir: str | Path,
    data_root: str | Path,
    train_transform: Optional[transforms.Compose] = None,
    eval_transform: Optional[transforms.Compose] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    class_to_idx: Optional[dict[str, int]] = None,
) -> dict[str, DataLoader]:
    """
    Create train/val/test dataloaders from manifest CSVs.
    Returns dict with keys 'train', 'val', 'test'.
    """
    manifests_dir = Path(manifests_dir)
    data_root = Path(data_root)

    # Build class_to_idx from training manifest to ensure consistency
    if class_to_idx is None:
        train_df = pd.read_csv(manifests_dir / 'train.csv')
        unique_classes = sorted(train_df['class_name'].unique())
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    dataloaders = {}

    splits = {
        'train': (manifests_dir / 'train.csv', train_transform, True),
        'val': (manifests_dir / 'val.csv', eval_transform, False),
        'test': (manifests_dir / 'test.csv', eval_transform, False),
    }

    for split_name, (manifest_path, transform, shuffle) in splits.items():
        if not manifest_path.exists():
            continue

        dataset = ManifestDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            transform=transform,
            class_to_idx=class_to_idx,
        )

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train'),
        )

    return dataloaders
