
import sys
import os
import torch
import math

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qinn.models.qil import QuantumInspiredLayer


def test_qil_forward_shape():
    """QIL produces correct output shape."""
    layer = QuantumInspiredLayer(16, 32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 32)


def test_qil_gradient_flow():
    """Gradients should flow through amplitude and phase branches."""
    layer = QuantumInspiredLayer(8, 8)
    x = torch.randn(4, 8, requires_grad=True)

    y = layer(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_phase_bounds():
    """Phase values must remain within [-pi, pi]."""
    layer = QuantumInspiredLayer(4, 4)
    x = torch.randn(10, 4)

    with torch.no_grad():
        phase_raw = layer.phase_proj(x)
        phase = torch.tanh(phase_raw) * math.pi

    assert torch.all(phase <= math.pi)
    assert torch.all(phase >= -math.pi)


def test_amplitude_stability():
    """Output should not contain NaN or Inf values."""
    layer = QuantumInspiredLayer(16, 16)
    x = torch.randn(32, 16) * 1000  # stress input

    y = layer(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_inference_mode():
    """Layer should behave correctly in eval mode."""
    layer = QuantumInspiredLayer(8, 8)
    layer.eval()

    x = torch.randn(4, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == (4, 8)
