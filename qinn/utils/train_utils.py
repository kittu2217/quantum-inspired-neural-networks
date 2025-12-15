# ============================================================
# Module: Training Utilities
# Author: Ghanta Krishna Murthy
# License: MIT
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """
    Train a model for one epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device identifier ("cpu" or "cuda").

    Returns:
        float: Average training loss.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[float, float]:
    """
    Evaluate a trained model.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation data loader.
        device (str): Device identifier ("cpu" or "cuda").

    Returns:
        Tuple[float, float]: (average loss, accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
