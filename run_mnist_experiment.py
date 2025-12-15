
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from qinn.models.qil import QuantumInspiredLayer
from qinn.utils.train_utils import train_one_epoch, evaluate

# -------------------------------
# Baseline Model
# -------------------------------
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
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
        self.fc1 = nn.Linear(28 * 28, 256)
        self.qil = QuantumInspiredLayer(256, 256)
        self.fc2 = nn.Linear(256, 10)

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
    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    baseline = BaselineMLP().to(device)
    qinn = QINN().to(device)

    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    opt_qinn = torch.optim.Adam(qinn.parameters(), lr=1e-3)

    epochs = 3

    print("\nStarting training...\n")

    for epoch in range(1, epochs + 1):
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
