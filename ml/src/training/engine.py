"""
Training engine: train_one_epoch and validate functions.
Core training loop logic separated from orchestration.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_max_norm: Optional[float] = 1.0,
) -> dict:
    """
    Train for one epoch.
    Returns dict with: loss, accuracy, num_samples, time_seconds.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if gradient_clip_max_norm is not None and gradient_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    elapsed = time.time() - start_time

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'num_samples': total,
        'time_seconds': round(elapsed, 2),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on validation/test set.
    Returns dict with: loss, accuracy, num_samples, predictions, labels.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    start_time = time.time()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    elapsed = time.time() - start_time

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'num_samples': total,
        'time_seconds': round(elapsed, 2),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> dict:
    """
    Measure single-image inference latency on given device.
    Returns dict with avg/min/max latency in milliseconds.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)

    # Measure
    latencies = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)  # ms

    return {
        'avg_ms': round(sum(latencies) / len(latencies), 2),
        'min_ms': round(min(latencies), 2),
        'max_ms': round(max(latencies), 2),
        'device': str(device),
    }
