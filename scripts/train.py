"""Training pipeline for the MNIST BNN.

Reproducible: seeded, clips latent weights to [-1, 1], saves the best
test-accuracy checkpoint together with the dataset mean/std (so export and
HW use the exact same standardization).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import get_config
from sw.model import BNN
from sw.training import get_dataloaders


def train(epochs_override=None, checkpoint_dir_override=None):
    cfg = get_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]
    dataset_dir = cfg["dataset_dir"]
    checkpoint_dir = Path(checkpoint_dir_override if checkpoint_dir_override is not None else train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, mean, std = get_dataloaders(dataset_dir, batch_size)

    model = BNN(
        input_size=model_cfg["input_size"],
        hidden_sizes=model_cfg["hidden_sizes"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Clip latent continuous weights to [-1, 1] (BNN convention).
            for p in model.parameters():
                if p.requires_grad:
                    p.data.clamp_(-1, 1)
            optimizer.step()
            running_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                correct += outputs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)
        test_acc = 100.0 * correct / total
        print(
            f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | Loss: {running_loss/len(train_loader):.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "mean": np.asarray(mean).reshape(-1).tolist(),
                    "std": np.asarray(std).reshape(-1).tolist(),
                },
                checkpoint_dir / "best_model.pth",
            )
            print(f"  -> Saved best model (acc={best_acc:.2f}%)")
        scheduler.step()

    print(f"Finished. Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    torch.manual_seed(get_config()["training"]["seed"])
    train(args.epochs)
