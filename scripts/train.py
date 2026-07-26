import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path

from config import get_config
from sw.model import BNN
from sw.training import get_dataloaders

def train(epochs_override=None):
    cfg = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = epochs_override if epochs_override is not None else cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    dataset_dir = cfg["dataset_dir"]
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(dataset_dir, batch_size)

    model = BNN(
        input_size=cfg["model"]["input_size"],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    # We use Square Hinge Loss commonly for BNNs or CrossEntropyLoss.
    # Let's try CrossEntropyLoss for simplicity. It works fine for BNNs.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optionally use a scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            # Optional: clip weights before optimizer step but BNN paper usually updates latent continuous weights
            # and clips them between [-1, 1].
            for p in model.parameters():
                if p.requires_grad:
                    p.grad.data.clamp_(-1, 1)

            optimizer.step()

            # Clip weights between [-1, 1]
            for p in model.parameters():
                if p.requires_grad:
                    p.data.clamp_(-1, 1)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print("  -> Saved new best model")

        scheduler.step()

    print(f"Finished Training. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    # Set seed
    cfg = get_config()
    torch.manual_seed(cfg["training"]["seed"])

    train(args.epochs)
