"""
train.py
--------
Contains training and validation loop functions.
Called each epoch from main.py.
"""

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full training epoch.

    Args:
        model     : the neural network
        loader    : training DataLoader
        optimizer : optimizer (Adam)
        criterion : loss function (CrossEntropyLoss)
        device    : 'cuda' or 'cpu'

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Runs one full validation pass (no gradients).

    Args:
        model     : the neural network
        loader    : val or test DataLoader
        criterion : loss function
        device    : 'cuda' or 'cpu'

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
