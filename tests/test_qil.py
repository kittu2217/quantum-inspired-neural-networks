
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from qinn.models.qil import QuantumInspiredLayer

def test_qil_forward_shape():
    layer = QuantumInspiredLayer(16, 32)
    x = torch.randn(4, 16)
    y = layer(x)

    assert y.shape == (4, 32)
