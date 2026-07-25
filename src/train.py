import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import json

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign().masked_fill(x == 0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1] = 0
        return grad_input

binarize = Binarize.apply

class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 3 * 3, 64, bias=False)
        self.bn4 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 10, bias=False)
        self.bn5 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = binarize(x)

        w1 = binarize(self.conv1.weight)
        x = F.conv2d(x, w1, padding=0)
        x = self.pool1(x)
        x = self.bn1(x)

        x = binarize(x)
        w2 = binarize(self.conv2.weight)
        x = F.conv2d(x, w2, padding=0)
        x = self.pool2(x)
        x = self.bn2(x)

        x = binarize(x)
        w3 = binarize(self.conv3.weight)
        x = F.conv2d(x, w3, padding=0)
        x = self.bn3(x)

        x = x.view(x.size(0), -1)

        x = binarize(x)
        w4 = binarize(self.fc1.weight)
        x = F.linear(x, w4)
        x = self.bn4(x)

        x = binarize(x)
        w5 = binarize(self.fc2.weight)
        x = F.linear(x, w5)
        x = self.bn5(x)

        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = BNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss() # Equivalent to sparse_categorical_crossentropy

    epochs = 12
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            for p in model.parameters():
                p.data.clamp_(-1, 1)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        train_acc = 100. * correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)

        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bnn_mnist.pth')

    # Export weights after training
    import export
    export.export_weights_mem(model, "mem_files")

if __name__ == "__main__":
    train()
