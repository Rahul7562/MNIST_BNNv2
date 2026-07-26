import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from sw.dataset import read_idx3_images, read_idx1_labels

class MNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform

        prefix = "train" if train else "t10k"
        images_path = self.data_dir / f"{prefix}-images.idx3-ubyte"
        labels_path = self.data_dir / f"{prefix}-labels.idx1-ubyte"

        self.images = read_idx3_images(images_path)
        self.labels = read_idx1_labels(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image shape is (28, 28) uint8
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor and float in range [0, 1] for transforms,
        # or transforms.ToTensor() expects PIL Image or numpy array

        if self.transform:
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.unsqueeze(0) # (1, 28, 28)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir, batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # Binary neural networks usually expect inputs between -1 and 1, or 0 and 1.
        # We will normalize to mean 0.5, std 0.5 which puts it in range [-1, 1].
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNISTDataset(data_dir, train=True, transform=train_transform)
    test_dataset = MNISTDataset(data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
