import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import random
import numpy as np

from sw.model.bnn import BNN
from sw.training.dataset import get_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, device, train_loader, optimizer, criterion, epoch, clip_val=1.0):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        model.clip_weights(clip_val)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()

    acc = 100. * correct / total
    loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}%")
    return acc, loss

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Epoch: {epoch} | Test Loss: {test_loss:.4f}  | Test Acc:  {acc:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_model', type=str, default='configs/model.yaml')
    parser.add_argument('--config_train', type=str, default='configs/training.yaml')
    args = parser.parse_args()

    with open(args.config_model, 'r') as f:
        config_model = yaml.safe_load(f)['model']

    with open(args.config_train, 'r') as f:
        config_train = yaml.safe_load(f)['training']

    set_seed(config_train['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config_train['checkpoint_dir'], exist_ok=True)

    train_loader, test_loader, mean, std = get_dataloaders(
        config_train['dataset_dir'],
        batch_size=config_train['batch_size']
    )

    model = BNN(
        input_size=config_model['input_size'],
        hidden_sizes=config_model['hidden_sizes'],
        num_classes=config_model['num_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_train['lr'])

    best_acc = 0.0
    best_model_path = os.path.join(config_train['checkpoint_dir'], 'best_model.pth')

    for epoch in range(1, config_train['epochs'] + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader, criterion, epoch)

        if acc > best_acc:
            best_acc = acc
            print(f"New best accuracy: {best_acc:.2f}%. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'mean': float(mean),
                'std': float(std)
            }, best_model_path)

    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    return best_acc

if __name__ == '__main__':
    main()
