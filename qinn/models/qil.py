
# ============================================================
# Quantum-Inspired Layer (QIL)
# ------------------------------------------------------------
# This layer mimics quantum superposition and interference
# using classical real-valued operations.
#
# Author: Ghanta Krishna Murthy
# ============================================================

import torch
import torch.nn as nn

class QuantumInspiredLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Linear projections for amplitude and phase
        self.amplitude_proj = nn.Linear(in_features, out_features)
        self.phase_proj = nn.Linear(in_features, out_features)

        # Normalization for stability
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        # Amplitude branch
        amplitude = self.amplitude_proj(x)

        # Phase branch
        phase = self.phase_proj(x)

        # Bound phase for numerical stability
        phase = torch.tanh(phase) * torch.pi

        # Quantum-inspired interference
        out = amplitude * torch.cos(phase)

        # Normalize output
        out = self.norm(out)

        return out
