# ============================================================
# Experiment: Noise Sensitivity Analysis (Trained Model)
# Author: Ghanta Krishna Murthy
# ============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import json
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from qinn.models.qil import QuantumInspiredLayer
from qinn.utils.train_utils import train_one_epoch


# -------------------------------
# Model Definition
# -------------------------------
class QINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.qil = QuantumInspiredLayer(256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.qil(x)
        return self.fc2(x)


def add_gaussian_noise(x, sigma):
    """Add Gaussian noise: x_noisy = x + N(0, sigma^2)"""
    return x + torch.randn_like(x) * sigma


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Data
    # -------------------------------
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # -------------------------------
    # Train Model
    # -------------------------------
    model = QINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training QINN for noise experiment...")
    for epoch in range(3):
        train_one_epoch(model, train_loader, optimizer, device)

    model.eval()

    # -------------------------------
    # Noise Sweep
    # -------------------------------
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    results = {}

    with torch.no_grad():
        for sigma in sigmas:
            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x_noisy = add_gaussian_noise(x, sigma)
                preds = model(x_noisy).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            acc = correct / total
            results[str(sigma)] = acc
            print(f"Sigma={sigma:.2f} → Accuracy={acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/noise_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Noise sweep results saved (trained model).")


if __name__ == "__main__":
    main()
