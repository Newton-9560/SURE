"""
utils/augmentations.py
----------------------
Augmentations for LLM hidden-state tensors (shape: (B, D), float32).

Both callables operate on batched tensors on whatever device they receive
and return a tensor of the same shape.

Weak aug:  small Gaussian noise (σ=0.02 × feature std)
Strong aug: feature dropout (p=0.2) + larger Gaussian noise (σ=0.1 × feature std)
"""

import torch
import torch.nn as nn


class WeakAugmentation(nn.Module):
    """Add small Gaussian noise scaled to the per-batch feature std.

    Args:
        noise_std: Noise scale as a fraction of the input's feature std.
    """

    def __init__(self, noise_std: float = 0.02):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # std computed over batch dim, shape (D,); clamp to avoid zero
        std = x.std(dim=0).clamp(min=1e-6)
        noise = torch.randn_like(x) * (self.noise_std * std)
        return x + noise


class StrongAugmentation(nn.Module):
    """Feature dropout followed by larger Gaussian noise.

    Args:
        dropout_p: Probability of zeroing out each feature dimension.
        noise_std: Noise scale as a fraction of the input's feature std.
    """

    def __init__(self, dropout_p: float = 0.2, noise_std: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute std before dropout so zeros don't bias it downward
        std = x.std(dim=0).clamp(min=1e-6)

        # Feature dropout: zero out each dim independently
        mask = torch.bernoulli(
            torch.full(x.shape, 1.0 - self.dropout_p, device=x.device)
        )
        x = x * mask

        # Larger noise scaled to feature std
        noise = torch.randn_like(x) * (self.noise_std * std)
        return x + noise


# Default instances ready to import
weak_aug   = WeakAugmentation(noise_std=0.02)
strong_aug = StrongAugmentation(dropout_p=0.2, noise_std=0.1)
