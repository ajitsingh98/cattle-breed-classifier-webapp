"""
Notebook generator: creates all 6 training notebooks as .ipynb files.
Run: python ml/scripts/generate_notebooks.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'notebooks'
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def save_notebook(nb, filename):
    path = NOTEBOOKS_DIR / filename
    with open(path, 'w') as f:
        nbformat.write(nb, f)
    print(f"  Created: {path}")


# ═══════════════════════════════════════════════════
# Notebook 0: Data Audit
# ═══════════════════════════════════════════════════
def create_00_data_audit():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell("# 00 — Data Audit\n\n"
            "Explore the Cattle Breed dataset before training:\n"
            "- Class distribution\n"
            "- Sample images per breed\n"
            "- Image size / resolution stats\n"
            "- Corrupt image check\n"
            "- Run stratified split and generate manifests"),

        new_code_cell(
            "import sys, os\n"
            "from pathlib import Path\n\n"
            "# Auto-detect project root\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from PIL import Image\n"
            "from collections import Counter\n\n"
            "%matplotlib inline\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "sns.set_palette('husl')"
        ),

        new_markdown_cell("## 1. Dataset Overview"),

        new_code_cell(
            "DATA_DIR = PROJECT_ROOT / 'Cattle_Resized'\n\n"
            "class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])\n"
            "print(f'Number of classes: {len(class_dirs)}')\n"
            "print(f'Class names:')\n"
            "for d in class_dirs:\n"
            "    count = len(list(d.glob('*')))\n"
            "    print(f'  {d.name}: {count} images')"
        ),

        new_markdown_cell("## 2. Class Distribution"),

        new_code_cell(
            "class_counts = {}\n"
            "for d in class_dirs:\n"
            "    class_counts[d.name] = len([f for f in d.iterdir() if f.is_file()])\n\n"
            "df_counts = pd.DataFrame([\n"
            "    {'breed': k, 'count': v} for k, v in class_counts.items()\n"
            "]).sort_values('count', ascending=True)\n\n"
            "fig, ax = plt.subplots(figsize=(10, 8))\n"
            "colors = ['#FF6B6B' if 'Buffalo' in b else '#4ECDC4' for b in df_counts['breed']]\n"
            "ax.barh(df_counts['breed'], df_counts['count'], color=colors)\n"
            "ax.set_xlabel('Number of Images')\n"
            "ax.set_title('Class Distribution', fontsize=14, fontweight='bold')\n\n"
            "# Add count labels\n"
            "for i, (_, row) in enumerate(df_counts.iterrows()):\n"
            "    ax.text(row['count'] + 1, i, str(row['count']), va='center', fontsize=9)\n\n"
            "plt.tight_layout()\n"
            "plt.savefig(str(PROJECT_ROOT / 'ml' / 'artifacts' / 'figures' / 'class_distribution.png'), dpi=150)\n"
            "plt.show()\n\n"
            "print(f'\\nTotal images: {sum(class_counts.values())}')\n"
            "print(f'Min class size: {min(class_counts.values())}')\n"
            "print(f'Max class size: {max(class_counts.values())}')\n"
            "print(f'Mean class size: {np.mean(list(class_counts.values())):.1f}')"
        ),

        new_markdown_cell("## 3. Sample Images per Breed"),

        new_code_cell(
            "fig, axes = plt.subplots(6, 5, figsize=(18, 22))\n"
            "axes = axes.flatten()\n\n"
            "for idx, class_dir in enumerate(class_dirs[:30]):\n"
            "    if idx >= len(axes):\n"
            "        break\n"
            "    images = sorted(class_dir.glob('*'))[:1]\n"
            "    if images:\n"
            "        img = Image.open(images[0]).convert('RGB')\n"
            "        axes[idx].imshow(img)\n"
            "    axes[idx].set_title(class_dir.name, fontsize=8, fontweight='bold')\n"
            "    axes[idx].axis('off')\n\n"
            "# Hide unused axes\n"
            "for idx in range(len(class_dirs), len(axes)):\n"
            "    axes[idx].axis('off')\n\n"
            "plt.suptitle('Sample Image per Breed', fontsize=14, fontweight='bold', y=1.01)\n"
            "plt.tight_layout()\n"
            "plt.savefig(str(PROJECT_ROOT / 'ml' / 'artifacts' / 'figures' / 'sample_images.png'), dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),

        new_markdown_cell("## 4. Image Size Statistics"),

        new_code_cell(
            "widths, heights, sizes_kb = [], [], []\n\n"
            "for class_dir in class_dirs:\n"
            "    for img_path in class_dir.iterdir():\n"
            "        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:\n"
            "            try:\n"
            "                with Image.open(img_path) as img:\n"
            "                    w, h = img.size\n"
            "                    widths.append(w)\n"
            "                    heights.append(h)\n"
            "                    sizes_kb.append(img_path.stat().st_size / 1024)\n"
            "            except Exception:\n"
            "                pass\n\n"
            "print(f'Total valid images: {len(widths)}')\n"
            "print(f'Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}')\n"
            "print(f'Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}')\n"
            "print(f'Size   — min: {min(sizes_kb):.1f}KB, max: {max(sizes_kb):.1f}KB, mean: {np.mean(sizes_kb):.1f}KB')\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "axes[0].hist(widths, bins=30, color='#4ECDC4', edgecolor='white')\n"
            "axes[0].set_title('Width Distribution')\n"
            "axes[1].hist(heights, bins=30, color='#FF6B6B', edgecolor='white')\n"
            "axes[1].set_title('Height Distribution')\n"
            "axes[2].hist(sizes_kb, bins=30, color='#45B7D1', edgecolor='white')\n"
            "axes[2].set_title('File Size (KB)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),

        new_markdown_cell("## 5. Validate & Split Dataset"),

        new_code_cell(
            "from ml.src.data.prepare_dataset import validate_images, stratified_split, \\\n"
            "    generate_class_balance_report, save_manifest\n"
            "from ml.src.utils.io import ensure_dirs\n\n"
            "# Create output directories\n"
            "manifests_dir = PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests'\n"
            "reports_dir = PROJECT_ROOT / 'ml' / 'artifacts' / 'reports'\n"
            "figures_dir = PROJECT_ROOT / 'ml' / 'artifacts' / 'figures'\n"
            "ensure_dirs(manifests_dir, reports_dir, figures_dir)\n\n"
            "# Validate\n"
            "valid_records, corrupt_records = validate_images(DATA_DIR)\n"
            "print(f'Valid: {len(valid_records)}, Corrupt: {len(corrupt_records)}')\n\n"
            "if corrupt_records:\n"
            "    print('\\nCorrupt files:')\n"
            "    for r in corrupt_records:\n"
            "        print(f\"  {r['image_path']}: {r['error']}\")"
        ),

        new_code_cell(
            "# Stratified split\n"
            "train, val, test = stratified_split(valid_records, 0.70, 0.15, 0.15, random_seed=42)\n\n"
            "for r in train: r['split'] = 'train'\n"
            "for r in val: r['split'] = 'val'\n"
            "for r in test: r['split'] = 'test'\n\n"
            "# Save manifests\n"
            "save_manifest(train, manifests_dir / 'train.csv')\n"
            "save_manifest(val, manifests_dir / 'val.csv')\n"
            "save_manifest(test, manifests_dir / 'test.csv')\n"
            "save_manifest(train + val + test, manifests_dir / 'all.csv')\n\n"
            "# Report\n"
            "report = generate_class_balance_report(train, val, test)\n"
            "for split in ['train', 'val', 'test']:\n"
            "    info = report[split]\n"
            "    print(f\"{split:>5}: {info['total']:>5} images | \"\n"
            "          f\"min={info['min_samples']} max={info['max_samples']} \"\n"
            "          f\"mean={info['mean_samples']}\")"
        ),

        new_markdown_cell("## 6. Breed Metadata Preview"),

        new_code_cell(
            "from ml.src.data.breed_metadata import BreedMetadataStore\n\n"
            "store = BreedMetadataStore()\n"
            "print(f'Loaded metadata for {store.num_breeds} breeds\\n')\n\n"
            "df_meta = pd.DataFrame(store.get_all())\n"
            "df_meta[['breed_name', 'animal_type', 'region', 'primary_use', 'avg_milk_liters_per_day']]"
        ),

        new_markdown_cell("---\n**✅ Data audit complete.** Manifests saved to `ml/artifacts/manifests/`. "
                          "Proceed to training notebooks."),
    ]
    save_notebook(nb, '00_data_audit.ipynb')


# ═══════════════════════════════════════════════════
# Notebook 1: MLP Baseline
# ═══════════════════════════════════════════════════
def create_01_mlp_baseline():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell(
            "# 01 — MLP Baseline\n\n"
            "Train a Multi-Layer Perceptron as a weak baseline.\n"
            "- Flatten 224×224×3 image → dense layers → classifier\n"
            "- **Not expected to win** — provides a lower bound for comparison\n\n"
            "### Expected Output\n"
            "- Best checkpoint → `ml/artifacts/checkpoints/mlp_best.pth`\n"
            "- Training curves, confusion matrix, per-class F1\n"
            "- Classification report and training summary JSON"
        ),

        new_code_cell(
            "import sys, os\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "from ml.src.utils.seed import set_seed\n"
            "from ml.src.utils.device import get_device\n"
            "from ml.src.utils.io import load_config\n"
            "from ml.src.utils.manifests import create_dataloaders\n"
            "from ml.src.data.transforms import get_train_transforms, get_eval_transforms\n"
            "from ml.src.models.mlp import CattleMLP\n"
            "from ml.src.training.trainer import Trainer\n\n"
            "set_seed(42)\n"
            "device = get_device()"
        ),

        new_markdown_cell("## 1. Load Config & Data"),

        new_code_cell(
            "config = load_config('mlp', PROJECT_ROOT / 'ml' / 'configs')\n"
            "config['num_classes'] = 26\n\n"
            "img_size = config['image']['size']\n"
            "batch_size = config['training']['batch_size']\n\n"
            "print(f'Image size: {img_size}')\n"
            "print(f'Batch size: {batch_size}')\n"
            "print(f'Epochs: {config[\"training\"][\"num_epochs\"]}')\n"
            "print(f'LR: {config[\"training\"][\"learning_rate\"]}')"
        ),

        new_code_cell(
            "train_transforms = get_train_transforms(img_size=img_size)\n"
            "eval_transforms = get_eval_transforms(img_size=img_size)\n\n"
            "dataloaders = create_dataloaders(\n"
            "    manifests_dir=PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests',\n"
            "    data_root=PROJECT_ROOT / 'Cattle_Resized',\n"
            "    train_transform=train_transforms,\n"
            "    eval_transform=eval_transforms,\n"
            "    batch_size=batch_size,\n"
            "    num_workers=config['data'].get('num_workers', 4),\n"
            ")\n\n"
            "class_names = dataloaders['train'].dataset.classes\n"
            "print(f'Classes: {len(class_names)}')\n"
            "print(f'Train: {len(dataloaders[\"train\"].dataset)}')\n"
            "print(f'Val: {len(dataloaders[\"val\"].dataset)}')\n"
            "print(f'Test: {len(dataloaders[\"test\"].dataset)}')"
        ),

        new_markdown_cell("## 2. Create Model"),

        new_code_cell(
            "model = CattleMLP.from_config(config)\n"
            "print(model)\n\n"
            "total_params = sum(p.numel() for p in model.parameters())\n"
            "print(f'\\nTotal parameters: {total_params:,}')\n"
            "print(f'Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)')"
        ),

        new_markdown_cell("## 3. Train"),

        new_code_cell(
            "trainer = Trainer(\n"
            "    model=model,\n"
            "    config=config,\n"
            "    dataloaders=dataloaders,\n"
            "    class_names=class_names,\n"
            "    device=device,\n"
            "    model_name='mlp',\n"
            ")\n\n"
            "training_summary = trainer.train()"
        ),

        new_markdown_cell("## 4. Evaluate on Test Set"),

        new_code_cell(
            "test_results = trainer.evaluate(split='test')"
        ),

        new_markdown_cell("## 5. Save Artifacts"),

        new_code_cell(
            "trainer.save_artifacts()\n\n"
            "print('\\n=== MLP Baseline Summary ===')\n"
            "print(f'Best Val Loss:  {training_summary[\"best_val_loss\"]:.4f}')\n"
            "print(f'Best Val Acc:   {training_summary[\"best_val_accuracy\"]:.4f}')\n"
            "print(f'Test Accuracy:  {test_results[\"metrics\"][\"accuracy\"]:.4f}')\n"
            "print(f'Test Macro F1:  {test_results[\"metrics\"][\"macro_f1\"]:.4f}')\n"
            "print(f'Latency:        {test_results[\"latency\"][\"avg_ms\"]:.2f} ms')\n"
            "print(f'Model Size:     {test_results[\"model_size_mb\"]:.2f} MB')"
        ),

        new_markdown_cell("## 6. Classification Report"),

        new_code_cell(
            "print(test_results['metrics']['classification_report'])"
        ),

        new_markdown_cell("---\n**✅ MLP Baseline complete.** Proceed to `02_cnn_from_scratch.ipynb`."),
    ]
    save_notebook(nb, '01_mlp_baseline.ipynb')


# ═══════════════════════════════════════════════════
# Notebook 2: CNN from Scratch
# ═══════════════════════════════════════════════════
def create_02_cnn_from_scratch():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell(
            "# 02 — CNN from Scratch\n\n"
            "Train a custom CNN with 5 convolutional blocks.\n"
            "- Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool (×5)\n"
            "- Global Average Pooling → FC classifier\n"
            "- Data augmentation: flip, rotation, jitter, crop\n\n"
            "### Expected Output\n"
            "- Best checkpoint → `ml/artifacts/checkpoints/cnn_best.pth`\n"
            "- Training curves, confusion matrix, per-class F1\n"
            "- Classification report and training summary JSON"
        ),

        new_code_cell(
            "import sys, os\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "from ml.src.utils.seed import set_seed\n"
            "from ml.src.utils.device import get_device\n"
            "from ml.src.utils.io import load_config\n"
            "from ml.src.utils.manifests import create_dataloaders\n"
            "from ml.src.data.transforms import get_train_transforms, get_eval_transforms\n"
            "from ml.src.models.cnn import CattleCNN\n"
            "from ml.src.training.trainer import Trainer\n\n"
            "set_seed(42)\n"
            "device = get_device()"
        ),

        new_markdown_cell("## 1. Load Config & Data"),

        new_code_cell(
            "config = load_config('cnn', PROJECT_ROOT / 'ml' / 'configs')\n"
            "config['num_classes'] = 26\n\n"
            "img_size = config['image']['size']\n"
            "batch_size = config['training']['batch_size']\n\n"
            "print(f'Image size: {img_size}')\n"
            "print(f'Batch size: {batch_size}')\n"
            "print(f'Conv channels: {config[\"model\"][\"architecture\"][\"conv_channels\"]}')\n"
            "print(f'Epochs: {config[\"training\"][\"num_epochs\"]}')"
        ),

        new_code_cell(
            "train_transforms = get_train_transforms(img_size=img_size)\n"
            "eval_transforms = get_eval_transforms(img_size=img_size)\n\n"
            "dataloaders = create_dataloaders(\n"
            "    manifests_dir=PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests',\n"
            "    data_root=PROJECT_ROOT / 'Cattle_Resized',\n"
            "    train_transform=train_transforms,\n"
            "    eval_transform=eval_transforms,\n"
            "    batch_size=batch_size,\n"
            "    num_workers=config['data'].get('num_workers', 4),\n"
            ")\n\n"
            "class_names = dataloaders['train'].dataset.classes\n"
            "print(f'Classes: {len(class_names)}')\n"
            "print(f'Train: {len(dataloaders[\"train\"].dataset)} | Val: {len(dataloaders[\"val\"].dataset)} | Test: {len(dataloaders[\"test\"].dataset)}')"
        ),

        new_markdown_cell("## 2. Create Model"),

        new_code_cell(
            "model = CattleCNN.from_config(config)\n"
            "print(model)\n\n"
            "total_params = sum(p.numel() for p in model.parameters())\n"
            "print(f'\\nTotal parameters: {total_params:,}')\n"
            "print(f'Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)')"
        ),

        new_markdown_cell("## 3. Train"),

        new_code_cell(
            "trainer = Trainer(\n"
            "    model=model,\n"
            "    config=config,\n"
            "    dataloaders=dataloaders,\n"
            "    class_names=class_names,\n"
            "    device=device,\n"
            "    model_name='cnn',\n"
            ")\n\n"
            "training_summary = trainer.train()"
        ),

        new_markdown_cell("## 4. Evaluate on Test Set"),

        new_code_cell("test_results = trainer.evaluate(split='test')"),

        new_markdown_cell("## 5. Save Artifacts"),

        new_code_cell(
            "trainer.save_artifacts()\n\n"
            "print('\\n=== CNN from Scratch Summary ===')\n"
            "print(f'Best Val Loss:  {training_summary[\"best_val_loss\"]:.4f}')\n"
            "print(f'Best Val Acc:   {training_summary[\"best_val_accuracy\"]:.4f}')\n"
            "print(f'Test Accuracy:  {test_results[\"metrics\"][\"accuracy\"]:.4f}')\n"
            "print(f'Test Macro F1:  {test_results[\"metrics\"][\"macro_f1\"]:.4f}')\n"
            "print(f'Latency:        {test_results[\"latency\"][\"avg_ms\"]:.2f} ms')\n"
            "print(f'Model Size:     {test_results[\"model_size_mb\"]:.2f} MB')"
        ),

        new_markdown_cell("## 6. Classification Report"),

        new_code_cell("print(test_results['metrics']['classification_report'])"),

        new_markdown_cell("---\n**✅ CNN from Scratch complete.** Proceed to `03_resnet_transfer_learning.ipynb`."),
    ]
    save_notebook(nb, '02_cnn_from_scratch.ipynb')


# ═══════════════════════════════════════════════════
# Notebook 3: ResNet Transfer Learning
# ═══════════════════════════════════════════════════
def create_03_resnet_transfer_learning():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell(
            "# 03 — ResNet50 Transfer Learning\n\n"
            "Fine-tune a pretrained ResNet50 (ImageNet) for cattle breed classification.\n\n"
            "### Two-Phase Training\n"
            "1. **Phase 1** (10 epochs): Freeze backbone, train only the classifier head\n"
            "2. **Phase 2** (20 epochs): Unfreeze `layer3` + `layer4`, fine-tune with lower LR\n\n"
            "### Expected Output\n"
            "- Best checkpoint → `ml/artifacts/checkpoints/resnet_best.pth`\n"
            "- Training curves, confusion matrix, per-class F1\n"
            "- Classification report and training summary JSON"
        ),

        new_code_cell(
            "import sys, os\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "from ml.src.utils.seed import set_seed\n"
            "from ml.src.utils.device import get_device\n"
            "from ml.src.utils.io import load_config\n"
            "from ml.src.utils.manifests import create_dataloaders\n"
            "from ml.src.data.transforms import get_train_transforms, get_eval_transforms\n"
            "from ml.src.models.resnet import CattleResNet\n"
            "from ml.src.training.trainer import Trainer\n\n"
            "set_seed(42)\n"
            "device = get_device()"
        ),

        new_markdown_cell("## 1. Load Config & Data"),

        new_code_cell(
            "config = load_config('resnet', PROJECT_ROOT / 'ml' / 'configs')\n"
            "config['num_classes'] = 26\n\n"
            "img_size = config['image']['size']\n"
            "batch_size = config['training']['batch_size']\n"
            "arch = config['model']['architecture']\n\n"
            "print(f'Backbone: {arch[\"backbone\"]}')\n"
            "print(f'Pretrained: {arch[\"pretrained\"]}')\n"
            "print(f'Freeze backbone: {arch[\"freeze_backbone\"]}')\n"
            "print(f'Unfreeze after: {arch[\"unfreeze_after_epochs\"]} epochs')\n"
            "print(f'Unfreeze layers: {arch[\"unfreeze_layers\"]}')\n"
            "print(f'Epochs: {config[\"training\"][\"num_epochs\"]}')\n"
            "print(f'LR: {config[\"training\"][\"learning_rate\"]}  Fine-tune LR: {config[\"training\"][\"fine_tune_lr\"]}')"
        ),

        new_code_cell(
            "train_transforms = get_train_transforms(img_size=img_size)\n"
            "eval_transforms = get_eval_transforms(img_size=img_size)\n\n"
            "dataloaders = create_dataloaders(\n"
            "    manifests_dir=PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests',\n"
            "    data_root=PROJECT_ROOT / 'Cattle_Resized',\n"
            "    train_transform=train_transforms,\n"
            "    eval_transform=eval_transforms,\n"
            "    batch_size=batch_size,\n"
            "    num_workers=config['data'].get('num_workers', 4),\n"
            ")\n\n"
            "class_names = dataloaders['train'].dataset.classes\n"
            "print(f'Classes: {len(class_names)}')\n"
            "print(f'Train: {len(dataloaders[\"train\"].dataset)} | Val: {len(dataloaders[\"val\"].dataset)} | Test: {len(dataloaders[\"test\"].dataset)}')"
        ),

        new_markdown_cell("## 2. Create Model"),

        new_code_cell(
            "model = CattleResNet.from_config(config)\n\n"
            "# Count trainable vs frozen params\n"
            "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
            "total = sum(p.numel() for p in model.parameters())\n"
            "print(f'Total parameters:     {total:,}')\n"
            "print(f'Trainable (Phase 1):  {trainable:,} ({100*trainable/total:.1f}%)')\n"
            "print(f'Frozen:               {total - trainable:,}')"
        ),

        new_markdown_cell("## 3. Two-Phase Training"),

        new_code_cell(
            "trainer = Trainer(\n"
            "    model=model,\n"
            "    config=config,\n"
            "    dataloaders=dataloaders,\n"
            "    class_names=class_names,\n"
            "    device=device,\n"
            "    model_name='resnet',\n"
            ")\n\n"
            "# Two-phase training: frozen backbone → partial fine-tuning\n"
            "training_summary = trainer.train_with_phase_switch(\n"
            "    phase1_epochs=arch['unfreeze_after_epochs'],\n"
            "    phase2_lr=config['training']['fine_tune_lr'],\n"
            "    unfreeze_layers=arch['unfreeze_layers'],\n"
            ")"
        ),

        new_markdown_cell("## 4. Evaluate on Test Set"),

        new_code_cell("test_results = trainer.evaluate(split='test')"),

        new_markdown_cell("## 5. Save Artifacts"),

        new_code_cell(
            "trainer.save_artifacts()\n\n"
            "print('\\n=== ResNet50 Transfer Learning Summary ===')\n"
            "print(f'Best Val Loss:  {training_summary[\"best_val_loss\"]:.4f}')\n"
            "print(f'Best Val Acc:   {training_summary[\"best_val_accuracy\"]:.4f}')\n"
            "print(f'Test Accuracy:  {test_results[\"metrics\"][\"accuracy\"]:.4f}')\n"
            "print(f'Test Macro F1:  {test_results[\"metrics\"][\"macro_f1\"]:.4f}')\n"
            "print(f'Latency:        {test_results[\"latency\"][\"avg_ms\"]:.2f} ms')\n"
            "print(f'Model Size:     {test_results[\"model_size_mb\"]:.2f} MB')"
        ),

        new_markdown_cell("## 6. Classification Report"),

        new_code_cell("print(test_results['metrics']['classification_report'])"),

        new_markdown_cell("---\n**✅ ResNet50 Transfer Learning complete.** Proceed to `04_vit_transfer_learning.ipynb`."),
    ]
    save_notebook(nb, '03_resnet_transfer_learning.ipynb')


# ═══════════════════════════════════════════════════
# Notebook 4: ViT Transfer Learning
# ═══════════════════════════════════════════════════
def create_04_vit_transfer_learning():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell(
            "# 04 — Vision Transformer (ViT-B/16) Transfer Learning\n\n"
            "Fine-tune a pretrained ViT-B/16 from `timm` for breed classification.\n\n"
            "### Two-Phase Training\n"
            "1. **Phase 1** (8 epochs): Freeze backbone, train custom head + warmup\n"
            "2. **Phase 2** (17 epochs): Unfreeze last 2 transformer blocks + norm, fine-tune\n\n"
            "### Requirements\n"
            "```bash\n"
            "pip install timm\n"
            "```\n\n"
            "### Expected Output\n"
            "- Best checkpoint → `ml/artifacts/checkpoints/vit_best.pth`\n"
            "- Training curves, confusion matrix, per-class F1"
        ),

        new_code_cell(
            "import sys, os\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "from ml.src.utils.seed import set_seed\n"
            "from ml.src.utils.device import get_device\n"
            "from ml.src.utils.io import load_config\n"
            "from ml.src.utils.manifests import create_dataloaders\n"
            "from ml.src.data.transforms import get_train_transforms, get_eval_transforms\n"
            "from ml.src.models.vit import CattleViT\n"
            "from ml.src.training.trainer import Trainer\n\n"
            "set_seed(42)\n"
            "device = get_device()"
        ),

        new_markdown_cell("## 1. Load Config & Data"),

        new_code_cell(
            "config = load_config('vit', PROJECT_ROOT / 'ml' / 'configs')\n"
            "config['num_classes'] = 26\n\n"
            "img_size = config['image']['size']\n"
            "batch_size = config['training']['batch_size']\n"
            "arch = config['model']['architecture']\n\n"
            "print(f'Backbone: {arch[\"backbone\"]}')\n"
            "print(f'Batch size: {batch_size} (smaller due to ViT memory)')\n"
            "print(f'Warmup: {config[\"training\"].get(\"warmup_epochs\", 0)} epochs')\n"
            "print(f'Epochs: {config[\"training\"][\"num_epochs\"]}')\n"
            "print(f'LR: {config[\"training\"][\"learning_rate\"]}  Fine-tune LR: {config[\"training\"][\"fine_tune_lr\"]}')"
        ),

        new_code_cell(
            "train_transforms = get_train_transforms(img_size=img_size)\n"
            "eval_transforms = get_eval_transforms(img_size=img_size)\n\n"
            "dataloaders = create_dataloaders(\n"
            "    manifests_dir=PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests',\n"
            "    data_root=PROJECT_ROOT / 'Cattle_Resized',\n"
            "    train_transform=train_transforms,\n"
            "    eval_transform=eval_transforms,\n"
            "    batch_size=batch_size,\n"
            "    num_workers=config['data'].get('num_workers', 4),\n"
            ")\n\n"
            "class_names = dataloaders['train'].dataset.classes\n"
            "print(f'Classes: {len(class_names)}')\n"
            "print(f'Train: {len(dataloaders[\"train\"].dataset)} | Val: {len(dataloaders[\"val\"].dataset)} | Test: {len(dataloaders[\"test\"].dataset)}')"
        ),

        new_markdown_cell("## 2. Create Model"),

        new_code_cell(
            "model = CattleViT.from_config(config)\n\n"
            "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
            "total = sum(p.numel() for p in model.parameters())\n"
            "print(f'Total parameters:     {total:,}')\n"
            "print(f'Trainable (Phase 1):  {trainable:,} ({100*trainable/total:.1f}%)')\n"
            "print(f'Frozen:               {total - trainable:,}')"
        ),

        new_markdown_cell("## 3. Two-Phase Training"),

        new_code_cell(
            "trainer = Trainer(\n"
            "    model=model,\n"
            "    config=config,\n"
            "    dataloaders=dataloaders,\n"
            "    class_names=class_names,\n"
            "    device=device,\n"
            "    model_name='vit',\n"
            ")\n\n"
            "training_summary = trainer.train_with_phase_switch(\n"
            "    phase1_epochs=arch['unfreeze_after_epochs'],\n"
            "    phase2_lr=config['training']['fine_tune_lr'],\n"
            "    unfreeze_layers=arch['unfreeze_layers'],\n"
            ")"
        ),

        new_markdown_cell("## 4. Evaluate on Test Set"),

        new_code_cell("test_results = trainer.evaluate(split='test')"),

        new_markdown_cell("## 5. Save Artifacts"),

        new_code_cell(
            "trainer.save_artifacts()\n\n"
            "print('\\n=== ViT-B/16 Transfer Learning Summary ===')\n"
            "print(f'Best Val Loss:  {training_summary[\"best_val_loss\"]:.4f}')\n"
            "print(f'Best Val Acc:   {training_summary[\"best_val_accuracy\"]:.4f}')\n"
            "print(f'Test Accuracy:  {test_results[\"metrics\"][\"accuracy\"]:.4f}')\n"
            "print(f'Test Macro F1:  {test_results[\"metrics\"][\"macro_f1\"]:.4f}')\n"
            "print(f'Latency:        {test_results[\"latency\"][\"avg_ms\"]:.2f} ms')\n"
            "print(f'Model Size:     {test_results[\"model_size_mb\"]:.2f} MB')"
        ),

        new_markdown_cell("## 6. Classification Report"),

        new_code_cell("print(test_results['metrics']['classification_report'])"),

        new_markdown_cell("---\n**✅ ViT Transfer Learning complete.** Proceed to `05_model_comparison.ipynb`."),
    ]
    save_notebook(nb, '04_vit_transfer_learning.ipynb')


# ═══════════════════════════════════════════════════
# Notebook 5: Model Comparison
# ═══════════════════════════════════════════════════
def create_05_model_comparison():
    nb = new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

    nb.cells = [
        new_markdown_cell(
            "# 05 — Model Comparison & Best Model Selection\n\n"
            "Compare all 4 trained models using a weighted composite score:\n\n"
            "| Metric | Weight |\n"
            "|--------|--------|\n"
            "| Macro F1 | 50% |\n"
            "| Top-1 Accuracy | 20% |\n"
            "| Inference Latency | 15% |\n"
            "| Model Size | 10% |\n"
            "| Calibration | 5% |\n\n"
            "### Expected Output\n"
            "- Radar chart comparing all models\n"
            "- Grouped bar chart of key metrics\n"
            "- Comparison table with rankings\n"
            "- Best model recommendation"
        ),

        new_code_cell(
            "import sys, os, json\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path(os.getcwd()).resolve()\n"
            "if 'notebooks' in str(PROJECT_ROOT):\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.parent\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print(f'Project root: {PROJECT_ROOT}')"
        ),

        new_code_cell(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "%matplotlib inline\n\n"
            "from ml.src.evaluation.compare_models import (\n"
            "    compute_weighted_scores,\n"
            "    plot_comparison_radar,\n"
            "    plot_comparison_bar,\n"
            "    generate_comparison_report,\n"
            ")"
        ),

        new_markdown_cell("## 1. Load All Model Reports"),

        new_code_cell(
            "reports_dir = PROJECT_ROOT / 'ml' / 'artifacts' / 'reports'\n"
            "model_names = ['mlp', 'cnn', 'resnet', 'vit']\n\n"
            "model_reports = []\n"
            "for name in model_names:\n"
            "    report_path = reports_dir / f'{name}_report.json'\n"
            "    if report_path.exists():\n"
            "        with open(report_path) as f:\n"
            "            report = json.load(f)\n"
            "        model_reports.append(report)\n"
            "        print(f'✓ Loaded {name} report')\n"
            "    else:\n"
            "        print(f'✗ Missing {name} report — run notebook {model_names.index(name)+1:02d} first')\n\n"
            "print(f'\\nLoaded {len(model_reports)} / {len(model_names)} model reports')"
        ),

        new_markdown_cell("## 2. Summary Table"),

        new_code_cell(
            "rows = []\n"
            "for r in model_reports:\n"
            "    metrics = r.get('metrics', {})\n"
            "    latency = r.get('latency', {})\n"
            "    rows.append({\n"
            "        'Model': r['model_name'],\n"
            "        'Accuracy': f\"{metrics.get('accuracy', 0):.4f}\",\n"
            "        'Macro F1': f\"{metrics.get('macro_f1', 0):.4f}\",\n"
            "        'Macro Precision': f\"{metrics.get('macro_precision', 0):.4f}\",\n"
            "        'Macro Recall': f\"{metrics.get('macro_recall', 0):.4f}\",\n"
            "        'Latency (ms)': f\"{latency.get('avg_ms', 0):.1f}\",\n"
            "        'Size (MB)': f\"{r.get('model_size_mb', 0):.1f}\",\n"
            "        'Parameters': f\"{r.get('num_parameters', 0):,}\",\n"
            "    })\n\n"
            "df_summary = pd.DataFrame(rows)\n"
            "df_summary.style.set_caption('Model Comparison Summary')"
        ),

        new_markdown_cell("## 3. Weighted Scoring & Ranking"),

        new_code_cell(
            "figures_dir = PROJECT_ROOT / 'ml' / 'artifacts' / 'figures' / 'comparison'\n"
            "figures_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "comparison_report = generate_comparison_report(\n"
            "    model_reports,\n"
            "    output_dir=figures_dir,\n"
            ")\n\n"
            "print('\\n=== Rankings ===')\n"
            "for r in comparison_report['rankings']:\n"
            "    print(f\"  #{r['rank']} {r['model_name']:>8s}  \"\n"
            "          f\"composite={r['composite_score']:.4f}  \"\n"
            "          f\"F1={r['raw_metrics']['macro_f1']:.4f}  \"\n"
            "          f\"Acc={r['raw_metrics']['accuracy']:.4f}  \"\n"
            "          f\"Lat={r['raw_metrics']['latency_ms']:.1f}ms  \"\n"
            "          f\"Size={r['raw_metrics']['model_size_mb']:.1f}MB\")"
        ),

        new_markdown_cell("## 4. Radar Chart"),

        new_code_cell(
            "from IPython.display import Image as IPImage, display\n"
            "radar_path = figures_dir / 'comparison_radar.png'\n"
            "if radar_path.exists():\n"
            "    display(IPImage(filename=str(radar_path), width=600))"
        ),

        new_markdown_cell("## 5. Bar Chart"),

        new_code_cell(
            "bar_path = figures_dir / 'comparison_bar.png'\n"
            "if bar_path.exists():\n"
            "    display(IPImage(filename=str(bar_path), width=800))"
        ),

        new_markdown_cell("## 6. Per-Model Confusion Matrices"),

        new_code_cell(
            "fig, axes = plt.subplots(1, len(model_reports), figsize=(6*len(model_reports), 5))\n"
            "if len(model_reports) == 1:\n"
            "    axes = [axes]\n\n"
            "for idx, report in enumerate(model_reports):\n"
            "    cm_path = PROJECT_ROOT / 'ml' / 'artifacts' / 'figures' / report['model_name'] / 'confusion_matrix.png'\n"
            "    if cm_path.exists():\n"
            "        img = plt.imread(str(cm_path))\n"
            "        axes[idx].imshow(img)\n"
            "        axes[idx].set_title(report['model_name'], fontweight='bold')\n"
            "    axes[idx].axis('off')\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),

        new_markdown_cell("## 7. Best Model Recommendation"),

        new_code_cell(
            "best = comparison_report['rankings'][0]\n"
            "print('=' * 60)\n"
            "print(f'🏆 RECOMMENDED MODEL: {best[\"model_name\"].upper()}')\n"
            "print('=' * 60)\n"
            "print(f'  Composite Score: {best[\"composite_score\"]:.4f}')\n"
            "print(f'  Macro F1:        {best[\"raw_metrics\"][\"macro_f1\"]:.4f}')\n"
            "print(f'  Accuracy:        {best[\"raw_metrics\"][\"accuracy\"]:.4f}')\n"
            "print(f'  Latency:         {best[\"raw_metrics\"][\"latency_ms\"]:.1f} ms')\n"
            "print(f'  Model Size:      {best[\"raw_metrics\"][\"model_size_mb\"]:.1f} MB')\n"
            "print('\\nCheckpoint path:')\n"
            "print(f'  ml/artifacts/checkpoints/{best[\"model_name\"]}_best.pth')\n"
            "print('\\nTo deploy this model, update the backend environment variable:')\n"
            "print(f'  MODEL_PATH=ml/artifacts/checkpoints/{best[\"model_name\"]}_best.pth')\n"
            "print(f'  MODEL_NAME={best[\"model_name\"]}')"
        ),

        new_markdown_cell(
            "## 8. Export for Deployment\n\n"
            "Copy the best checkpoint to the `models/` directory for the backend to load."
        ),

        new_code_cell(
            "import shutil\n\n"
            "best_name = best['model_name']\n"
            "src = PROJECT_ROOT / 'ml' / 'artifacts' / 'checkpoints' / f'{best_name}_best.pth'\n"
            "dst = PROJECT_ROOT / 'models' / f'{best_name}_best.pth'\n\n"
            "if src.exists():\n"
            "    shutil.copy2(src, dst)\n"
            "    print(f'✓ Copied {src.name} → {dst}')\n\n"
            "    # Also save class names for the backend\n"
            "    import torch\n"
            "    ckpt = torch.load(src, map_location='cpu', weights_only=False)\n"
            "    if 'class_names' in ckpt:\n"
            "        classes_path = PROJECT_ROOT / 'models' / 'classes.txt'\n"
            "        with open(classes_path, 'w') as f:\n"
            "            f.write(','.join(ckpt['class_names']))\n"
            "        print(f'✓ Updated {classes_path}')\n"
            "else:\n"
            "    print(f'✗ Checkpoint not found: {src}')"
        ),

        new_markdown_cell(
            "---\n"
            "**✅ Model comparison complete!**\n\n"
            "### Next Steps\n"
            "1. Start the backend: `PYTHONPATH=. uvicorn backend.app.main:app --reload`\n"
            "2. Start the frontend: `cd frontend && npm run dev`\n"
            "3. Upload images and verify predictions"
        ),
    ]
    save_notebook(nb, '05_model_comparison.ipynb')


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating notebooks...\n")
    create_00_data_audit()
    create_01_mlp_baseline()
    create_02_cnn_from_scratch()
    create_03_resnet_transfer_learning()
    create_04_vit_transfer_learning()
    create_05_model_comparison()
    print(f"\n✅ All 6 notebooks created in {NOTEBOOKS_DIR}")
