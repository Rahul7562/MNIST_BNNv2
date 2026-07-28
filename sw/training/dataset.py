"""MNIST data loading and preprocessing for the BNN rebuild.

Contract: docs/ARCHITECTURE.md §4. The dataset supplies REAL-VALUED,
per-pixel standardized pixels in [~ -? , +? ]. The MODEL binarizes the
input by sign (see sw/model/bnn.py). This keeps a single, identical
standardization path for software, convert_image, and the HW input.

Standardization uses the training-set per-pixel mean/std (computed once and
persisted in the checkpoint / export_meta so HW and SW agree exactly).
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def read_idx3(filename):
    with open(filename, "rb") as f:
        _, _, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def read_idx1(filename):
    with open(filename, "rb") as f:
        _, _, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


class MNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, mean=None, std=None):
        self.root_dir = root_dir
        self.train = train
        prefix = "train" if train else "t10k"
        self.images = read_idx3(os.path.join(root_dir, f"{prefix}-images.idx3-ubyte"))
        self.labels = read_idx1(os.path.join(root_dir, f"{prefix}-labels.idx1-ubyte"))
        self.images = self.images.astype(np.float32) / 255.0  # [0, 1]
        # Per-pixel mean/std over the training set (flattened to 784).
        if mean is None or std is None:
            self.mean = self.images.reshape(self.images.shape[0], -1).mean(axis=0)
            self.std = self.images.reshape(self.images.shape[0], -1).std(axis=0) + 1e-7
        else:
            self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
            self.std = np.asarray(std, dtype=np.float32).reshape(-1) + 1e-7

    def set_stats(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32) + 1e-7

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(-1)  # (784,)
        # Standardize per pixel: x' = (x - mean) / std. REAL-VALUED (model binarizes).
        x = (img - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


def get_dataloaders(dataset_dir, batch_size=256, num_workers=0):
    """num_workers=0 by default (fork-servers can be flaky on some hosts)."""
    train_dataset = MNISTDataset(dataset_dir, train=True)
    test_dataset = MNISTDataset(
        dataset_dir, train=False, mean=train_dataset.mean, std=train_dataset.std
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, train_dataset.mean, train_dataset.std
