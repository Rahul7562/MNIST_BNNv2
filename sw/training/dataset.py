import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def read_idx3(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def read_idx1(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

class MNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train

        if self.train:
            self.images = read_idx3(os.path.join(root_dir, 'train-images.idx3-ubyte'))
            self.labels = read_idx1(os.path.join(root_dir, 'train-labels.idx1-ubyte'))
        else:
            self.images = read_idx3(os.path.join(root_dir, 't10k-images.idx3-ubyte'))
            self.labels = read_idx1(os.path.join(root_dir, 't10k-labels.idx1-ubyte'))

        self.images = self.images.astype(np.float32) / 255.0

        self.mean = self.images.mean()
        self.std = self.images.std()

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = img.reshape(-1)

        img = (img - self.mean) / (self.std + 1e-7)

        # Binarize to {-1, +1}
        img = np.where(img >= 0, 1.0, -1.0).astype(np.float32)

        return torch.tensor(img), torch.tensor(label, dtype=torch.long)

def get_dataloaders(dataset_dir, batch_size=100, num_workers=2):
    train_dataset = MNISTDataset(dataset_dir, train=True)
    test_dataset = MNISTDataset(dataset_dir, train=False)

    test_dataset.set_stats(train_dataset.mean, train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.mean, train_dataset.std
