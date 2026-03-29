"""
Visualization module.
Generates training curves, confusion matrices, and per-class metric charts.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# Style defaults
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'train': '#2196F3',
    'val': '#FF5722',
    'primary': '#4CAF50',
    'secondary': '#FFC107',
}


def plot_training_curves(
    history: dict,
    save_path: str | Path = None,
    title: str = 'Training Curves',
    figsize: tuple = (14, 5),
) -> None:
    """Plot loss and accuracy curves for training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], color=COLORS['train'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], color=COLORS['val'], label='Validation', linewidth=2)
    axes[0].set_title(f'{title} - Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_accuracy'], color=COLORS['train'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], color=COLORS['val'], label='Validation', linewidth=2)
    axes[1].set_title(f'{title} - Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training curves to {save_path}")

    plt.close()


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str | Path = None,
    title: str = 'Confusion Matrix',
    figsize: tuple = None,
    normalize: bool = True,
) -> None:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred)
    n_classes = len(class_names)

    if figsize is None:
        size = max(10, n_classes * 0.5)
        figsize = (size, size)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix to {save_path}")

    plt.close()


def plot_per_class_f1(
    per_class_metrics: dict,
    class_names: list[str],
    save_path: str | Path = None,
    title: str = 'Per-Class F1 Score',
    figsize: tuple = None,
) -> None:
    """Plot horizontal bar chart of per-class F1 scores."""
    f1_scores = []
    names = []

    for name in class_names:
        if name in per_class_metrics:
            f1_scores.append(per_class_metrics[name]['f1'])
            names.append(name)

    # Sort by F1 score
    sorted_pairs = sorted(zip(names, f1_scores), key=lambda x: x[1])
    names, f1_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

    n = len(names)
    if figsize is None:
        figsize = (10, max(6, n * 0.35))

    fig, ax = plt.subplots(figsize=figsize)

    colors = [plt.cm.RdYlGn(score) for score in f1_scores]

    bars = ax.barh(range(n), f1_scores, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)

    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.2f}', va='center', fontsize=8)

    ax.axvline(x=np.mean(list(f1_scores)), color='red', linestyle='--',
               alpha=0.7, label=f'Mean: {np.mean(list(f1_scores)):.2f}')
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved per-class F1 chart to {save_path}")

    plt.close()


def plot_sample_predictions(
    images: list,
    true_labels: list[str],
    pred_labels: list[str],
    confidences: list[float],
    save_path: str | Path = None,
    title: str = 'Sample Predictions',
    n_samples: int = 16,
) -> None:
    """Plot grid of sample predictions with true vs predicted labels."""
    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if hasattr(img, 'numpy'):
            img = img.numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        correct = true_labels[i] == pred_labels[i]
        color = 'green' if correct else 'red'
        ax.set_title(
            f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.0%})",
            fontsize=8, color=color, fontweight='bold',
        )
        ax.axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()
