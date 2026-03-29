"""
Dataset preparation pipeline.
- Validates images (opens each, removes corrupt)
- Performs stratified train/val/test split (70/15/15)
- Generates manifest CSVs and class balance report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def validate_images(data_dir: Path, verbose: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Walk through all class folders and validate each image.
    Returns (valid_records, corrupt_records).
    """
    valid_records = []
    corrupt_records = []

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        ])

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open to get actual size (verify closes the file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode

                valid_records.append({
                    'image_path': str(img_path.relative_to(data_dir)),
                    'class_name': class_name,
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'file_size_kb': round(img_path.stat().st_size / 1024, 1),
                })
            except Exception as e:
                corrupt_records.append({
                    'image_path': str(img_path.relative_to(data_dir)),
                    'class_name': class_name,
                    'error': str(e),
                })
                if verbose:
                    print(f"  [CORRUPT] {img_path}: {e}")

    if verbose:
        print(f"\nValidation complete: {len(valid_records)} valid, {len(corrupt_records)} corrupt")

    return valid_records, corrupt_records


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Perform stratified split into train/val/test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    df = pd.DataFrame(records)
    labels = df['class_name'].values

    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        range(len(df)),
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed,
    )

    # Second split: val vs test
    temp_labels = labels[temp_idx]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=random_seed,
    )

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    return train_records, val_records, test_records


def generate_class_balance_report(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
) -> dict:
    """Generate class distribution statistics for each split."""
    report = {}

    for split_name, records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        counter = Counter(r['class_name'] for r in records)
        total = len(records)
        report[split_name] = {
            'total': total,
            'num_classes': len(counter),
            'class_distribution': dict(sorted(counter.items())),
            'min_samples': min(counter.values()) if counter else 0,
            'max_samples': max(counter.values()) if counter else 0,
            'mean_samples': round(total / len(counter), 1) if counter else 0,
        }

    return report


def save_manifest(records: list[dict], output_path: Path) -> None:
    """Save records as a CSV manifest."""
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset: validate, split, and generate manifests")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(PROJECT_ROOT / 'Cattle_Resized'),
        help='Path to the raw image dataset (folder-per-class)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'ml' / 'artifacts' / 'manifests'),
        help='Output directory for manifest CSVs'
    )
    parser.add_argument(
        '--reports-dir',
        type=str,
        default=str(PROJECT_ROOT / 'ml' / 'artifacts' / 'reports'),
        help='Output directory for reports'
    )
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate images
    print("=" * 60)
    print("Step 1: Validating images...")
    print("=" * 60)
    valid_records, corrupt_records = validate_images(data_dir)

    # Save corrupt image report
    if corrupt_records:
        corrupt_path = reports_dir / 'corrupt_images.json'
        with open(corrupt_path, 'w') as f:
            json.dump(corrupt_records, f, indent=2)
        print(f"  Corrupt image report saved to {corrupt_path}")

    # Step 2: Stratified split
    print("\n" + "=" * 60)
    print("Step 2: Performing stratified split...")
    print("=" * 60)
    train_records, val_records, test_records = stratified_split(
        valid_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    # Add split labels
    for r in train_records:
        r['split'] = 'train'
    for r in val_records:
        r['split'] = 'val'
    for r in test_records:
        r['split'] = 'test'

    print(f"  Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    # Step 3: Save manifests
    print("\n" + "=" * 60)
    print("Step 3: Saving manifests...")
    print("=" * 60)
    save_manifest(train_records, output_dir / 'train.csv')
    save_manifest(val_records, output_dir / 'val.csv')
    save_manifest(test_records, output_dir / 'test.csv')
    save_manifest(train_records + val_records + test_records, output_dir / 'all.csv')

    # Step 4: Generate class balance report
    print("\n" + "=" * 60)
    print("Step 4: Generating class balance report...")
    print("=" * 60)
    balance_report = generate_class_balance_report(train_records, val_records, test_records)

    report_path = reports_dir / 'class_balance_report.json'
    with open(report_path, 'w') as f:
        json.dump(balance_report, f, indent=2)
    print(f"  Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    for split_name in ['train', 'val', 'test']:
        info = balance_report[split_name]
        print(f"  {split_name:>5}: {info['total']:>5} images | "
              f"{info['num_classes']} classes | "
              f"min={info['min_samples']} max={info['max_samples']} "
              f"mean={info['mean_samples']}")

    print("\nDone!")


if __name__ == '__main__':
    main()
