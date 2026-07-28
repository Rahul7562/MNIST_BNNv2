"""Classify a user-drawn digit saved as `my_digit.png` (or any image path).

Pipeline (must match training/export exactly):
  PNG -> grayscale -> 28x28 -> flatten -> per-pixel standardize with the
  checkpoint mean/std -> binarize by sign -> BNN (SW) and the HW popcount
  recompute (from mem_files). Prints both predictions.

Usage:
  python3 scripts/convert_image.py [path.png]   # default: my_digit.png
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sw.model.bnn import BNN


def load_and_preprocess(path, mean, std):
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.reshape(-1)
    x_norm = (arr - mean) / (std + 1e-7)
    return x_norm


def hw_popcount_inference(x_bits, mem_dir, meta):
    def _read(name):
        with open(os.path.join(mem_dir, name)) as f:
            return [l.strip() for l in f if l.strip()]

    def _bits(h, w):
        return np.array([int(b) for b in format(int(h, 16), f"0{w}b")], dtype=np.int32)

    w1 = _read("layer1_weights.mem"); th1 = np.loadtxt(os.path.join(mem_dir, "layer1_thresholds.mem"))
    w2 = _read("layer2_weights.mem"); th2 = np.loadtxt(os.path.join(mem_dir, "layer2_thresholds.mem"))
    w3 = _read("layer3_weights.mem"); off3 = np.loadtxt(os.path.join(mem_dir, "layer3_offsets.mem"))
    scale3 = np.loadtxt(os.path.join(mem_dir, "layer3_scales.mem"))
    W1 = np.array([_bits(h, meta["layer_shapes"][0][0]) for h in w1], dtype=np.int32)
    W2 = np.array([_bits(h, meta["layer_shapes"][1][0]) for h in w2], dtype=np.int32)
    W3 = np.array([_bits(h, meta["layer_shapes"][2][0]) for h in w3], dtype=np.int32)

    a = x_bits
    p1 = (1 - (W1 ^ a)).sum(1); a = (p1 >= th1).astype(np.int32)
    p2 = (1 - (W2 ^ a)).sum(1); a = (p2 >= th2).astype(np.int32)
    p3 = (1 - (W3 ^ a)).sum(1); z3 = 2 * p3 - W3.shape[1]
    logits = scale3 * z3 + off3
    return int(np.argmax(logits))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "my_digit.png")
    if not os.path.exists(path):
        print(f"ERROR: image not found at {path}")
        sys.exit(1)

    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "model.yaml")))
    ckpt = torch.load(
        os.path.join(REPO, "checkpoints", "best_model.pth"), map_location="cpu", weights_only=False
    )
    mean = np.asarray(ckpt["mean"]).reshape(-1)
    std = np.asarray(ckpt["std"]).reshape(-1)
    model = BNN(input_size=cfg["input_size"], hidden_sizes=cfg["hidden_sizes"], num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_norm = load_and_preprocess(path, mean, std)
    x_bits = ((np.where(x_norm >= 0, 1.0, -1.0) + 1) / 2).astype(np.int32)

    with torch.no_grad():
        sw_pred = int(model(torch.tensor(x_norm.reshape(1, -1))).argmax(dim=1).item())
    meta = json.load(open(os.path.join(REPO, "mem_files", "export_meta.json")))
    hw_pred = hw_popcount_inference(x_bits, os.path.join(REPO, "mem_files"), meta)

    print(f"Image: {path}")
    print(f"  SW model prediction : {sw_pred}")
    print(f"  HW popcount prediction: {hw_pred}")
    if sw_pred != hw_pred:
        print("  WARNING: SW and HW popcount disagree (should be rare; check artifact sync).")
    else:
        print("  OK: SW and HW popcount agree.")


if __name__ == "__main__":
    main()
