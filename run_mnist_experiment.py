# ============================================================
# End-to-End MNIST Experiment (Config-Driven & Reproducible)
# Author: Ghanta Krishna Murthy
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from qinn.models.qil import QuantumInspiredLayer
from qinn.utils.train_utils import train_one_epoch, evaluate

# -------------------------------
# Load configuration
# -------------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["experiment"]["seed"]
EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]
LR = cfg["training"]["learning_rate"]
HIDDEN_DIM = cfg["model"]["hidden_dim"]

# -------------------------------
# Reproducibility
# -------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------
# Baseline Model
# -------------------------------
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# -------------------------------
# Quantum-Inspired Model
# -------------------------------
class QINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, HIDDEN_DIM)
        self.qil = QuantumInspiredLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.qil(x)
        return self.fc2(x)

# -------------------------------
# Main Experiment
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device        : {device}")
    print(f"Seed          : {SEED}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs       : {EPOCHS}")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    baseline = BaselineMLP().to(device)
    qinn = QINN().to(device)

    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=LR)
    opt_qinn = torch.optim.Adam(qinn.parameters(), lr=LR)

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(baseline, train_loader, opt_baseline, device)
        train_one_epoch(qinn, train_loader, opt_qinn, device)

        _, bl_acc = evaluate(baseline, test_loader, device)
        _, qn_acc = evaluate(qinn, test_loader, device)

        print(f"Epoch {epoch}")
        print(f"  Baseline Test Accuracy: {bl_acc:.4f}")
        print(f"  QINN     Test Accuracy: {qn_acc:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
