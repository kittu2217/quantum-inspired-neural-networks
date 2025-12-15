# ============================================================
# Module: Quantum-Inspired Layer (QIL)
# Author: Ghanta Krishna Murthy
# License: MIT
# ============================================================

import torch
import torch.nn as nn
from torch import Tensor


class QuantumInspiredLayer(nn.Module):
    """
    Quantum-Inspired Layer (QIL).

    This layer mimics quantum interference by decomposing features into
    amplitude and phase components and recombining them via cosine-based
    interference.

    The implementation is fully classical and GPU-friendly.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the Quantum-Inspired Layer.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
        """
        super().__init__()

        self.amplitude_proj = nn.Linear(in_features, out_features)
        self.phase_proj = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Quantum-Inspired Layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        amplitude = self.amplitude_proj(x)
        phase = self.phase_proj(x)

        # Bound phase to [-pi, pi] for stability
        phase = torch.tanh(phase) * torch.pi

        out = amplitude * torch.cos(phase)
        out = self.norm(out)

        return out
