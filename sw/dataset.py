import numpy as np
import struct
from pathlib import Path

def read_idx3_images(filepath: str | Path) -> np.ndarray:
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for IDX3 images")

        # Read the rest of the file into a numpy array
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)

        # Reshape to (num_images, rows, cols)
        return data.reshape(num_images, rows, cols)

def read_idx1_labels(filepath: str | Path) -> np.ndarray:
    with open(filepath, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for IDX1 labels")

        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8)
