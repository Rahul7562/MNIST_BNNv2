import sys
from pathlib import Path

import config
from sw import dataset

def main():
    cfg = config.get_config()
    dataset_dir = Path(cfg["dataset_dir"])

    train_images_path = dataset_dir / "train-images.idx3-ubyte"
    train_labels_path = dataset_dir / "train-labels.idx1-ubyte"
    test_images_path = dataset_dir / "t10k-images.idx3-ubyte"
    test_labels_path = dataset_dir / "t10k-labels.idx1-ubyte"

    try:
        train_images = dataset.read_idx3_images(train_images_path)
        train_labels = dataset.read_idx1_labels(train_labels_path)
        test_images = dataset.read_idx3_images(test_images_path)
        test_labels = dataset.read_idx1_labels(test_labels_path)

        print("MNIST Dataset Verification:")
        print("-" * 25)
        print(f"Train images shape: {train_images.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test images shape:  {test_images.shape}")
        print(f"Test labels shape:  {test_labels.shape}")

        assert train_images.shape == (60000, 28, 28), "Invalid train images shape"
        assert train_labels.shape == (60000,), "Invalid train labels shape"
        assert test_images.shape == (10000, 28, 28), "Invalid test images shape"
        assert test_labels.shape == (10000,), "Invalid test labels shape"

        print("\nDataset successfully loaded and verified!")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
