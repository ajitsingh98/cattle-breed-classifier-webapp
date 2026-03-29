"""
Model comparison module.
Applies weighted scoring to select the best model.
Weights: 50% F1 + 20% accuracy + 15% latency + 10% size + 5% calibration.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Scoring weights
DEFAULT_WEIGHTS = {
    'macro_f1': 0.50,
    'accuracy': 0.20,
    'latency': 0.15,
    'model_size': 0.10,
    'calibration': 0.05,
}


def normalize_metric(values: list[float], higher_is_better: bool = True) -> list[float]:
    """Min-max normalize metrics to [0, 1] range."""
    if not values:
        return values
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - min_val) / (max_val - min_val) for v in values]
    else:
        return [(max_val - v) / (max_val - min_val) for v in values]


def compute_weighted_scores(
    model_reports: list[dict],
    weights: Optional[dict] = None,
) -> list[dict]:
    """
    Compute weighted composite scores for model comparison.

    Args:
        model_reports: list of per-model evaluation reports
        weights: scoring weights dict

    Returns:
        list of scored models, sorted by composite score (descending)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Extract raw metrics
    model_names = []
    f1_scores = []
    accuracies = []
    latencies = []
    sizes = []

    for report in model_reports:
        model_names.append(report['model_name'])
        metrics = report.get('metrics', {})
        f1_scores.append(metrics.get('macro_f1', 0))
        accuracies.append(metrics.get('accuracy', 0))
        latencies.append(report.get('latency', {}).get('avg_ms', 100))
        sizes.append(report.get('model_size_mb', 100))

    # Normalize
    norm_f1 = normalize_metric(f1_scores, higher_is_better=True)
    norm_acc = normalize_metric(accuracies, higher_is_better=True)
    norm_lat = normalize_metric(latencies, higher_is_better=False)
    norm_size = normalize_metric(sizes, higher_is_better=False)

    # Compute composite score
    results = []
    for i, name in enumerate(model_names):
        composite = (
            weights['macro_f1'] * norm_f1[i]
            + weights['accuracy'] * norm_acc[i]
            + weights['latency'] * norm_lat[i]
            + weights['model_size'] * norm_size[i]
            # calibration placeholder (set to 0.5 if not provided)
            + weights['calibration'] * 0.5
        )

        results.append({
            'model_name': name,
            'composite_score': round(composite, 4),
            'raw_metrics': {
                'macro_f1': f1_scores[i],
                'accuracy': accuracies[i],
                'latency_ms': latencies[i],
                'model_size_mb': sizes[i],
            },
            'normalized_metrics': {
                'macro_f1': round(norm_f1[i], 4),
                'accuracy': round(norm_acc[i], 4),
                'latency': round(norm_lat[i], 4),
                'model_size': round(norm_size[i], 4),
            },
            'rank': 0,  # filled below
        })

    # Sort and rank
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


def plot_comparison_radar(
    scored_models: list[dict],
    save_path: str | Path = None,
    title: str = 'Model Comparison',
) -> None:
    """Plot radar chart comparing models on normalized metrics."""
    categories = ['F1 Score', 'Accuracy', 'Speed', 'Compactness']
    n_categories = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = [n / float(n_categories) * 2 * np.pi for n in range(n_categories)]
    angles += angles[:1]  # Complete the circle

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for idx, model in enumerate(scored_models):
        norm = model['normalized_metrics']
        values = [
            norm['macro_f1'],
            norm['accuracy'],
            norm['latency'],
            norm['model_size'],
        ]
        values += values[:1]

        color = colors[idx % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, color=color,
                label=f"{model['model_name']} ({model['composite_score']:.3f})")
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved radar chart to {save_path}")

    plt.close()


def plot_comparison_bar(
    scored_models: list[dict],
    save_path: str | Path = None,
    title: str = 'Model Comparison - Key Metrics',
) -> None:
    """Plot grouped bar chart comparing raw metrics across models."""
    model_names = [m['model_name'] for m in scored_models]
    metrics_to_plot = {
        'Macro F1': [m['raw_metrics']['macro_f1'] for m in scored_models],
        'Accuracy': [m['raw_metrics']['accuracy'] for m in scored_models],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # F1 and Accuracy
    x = np.arange(len(model_names))
    width = 0.35

    axes[0].bar(x - width/2, metrics_to_plot['Macro F1'], width, label='Macro F1', color='#2196F3')
    axes[0].bar(x + width/2, metrics_to_plot['Accuracy'], width, label='Accuracy', color='#4CAF50')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classification Performance')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Latency
    latencies = [m['raw_metrics']['latency_ms'] for m in scored_models]
    axes[1].bar(model_names, latencies, color='#FF5722')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Inference Latency')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    # Model Size
    sizes = [m['raw_metrics']['model_size_mb'] for m in scored_models]
    axes[2].bar(model_names, sizes, color='#9C27B0')
    axes[2].set_ylabel('Size (MB)')
    axes[2].set_title('Model Size')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved comparison bar chart to {save_path}")

    plt.close()


def generate_comparison_report(
    model_reports: list[dict],
    output_dir: str | Path = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Full comparison pipeline:
    1. Compute weighted scores
    2. Generate radar chart
    3. Generate bar chart
    4. Save JSON report
    5. Return best model recommendation
    """
    scored = compute_weighted_scores(model_reports, weights)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plots
        plot_comparison_radar(scored, output_dir / 'comparison_radar.png')
        plot_comparison_bar(scored, output_dir / 'comparison_bar.png')

        # Report JSON
        report = {
            'weights_used': weights or DEFAULT_WEIGHTS,
            'rankings': scored,
            'best_model': scored[0]['model_name'],
            'recommendation': (
                f"Recommended model: {scored[0]['model_name']} "
                f"(composite score: {scored[0]['composite_score']:.4f})"
            ),
        }
        with open(output_dir / 'comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Best model: {scored[0]['model_name']} "
              f"(score: {scored[0]['composite_score']:.4f})")

        return report

    return {'rankings': scored, 'best_model': scored[0]['model_name']}
