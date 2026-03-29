"""
Device detection and management utilities.
"""

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Auto-detect the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def get_device_info() -> dict:
    """Return device information as a dictionary."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'device': str(get_device()),
    }
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_memory_gb'] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    return info
