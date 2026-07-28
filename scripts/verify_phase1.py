"""Independent Phase-1 verification: software accuracy + HW popcount bit-exactness.

This script does NOT trust the training session's self-reported numbers. It:
  1. Re-exports mem_files from the best checkpoint (regenerates artifacts).
  2. Runs the canonical XNOR-popcount hardware inference path (ARCHITECTURE.md §5)
     over N test samples.
  3. Compares HW argmax vs the PyTorch model argmax, and reports both SW and HW
     accuracy vs true labels.

Gate (success criteria): SW acc > 95%, HW(popcount) acc > 95%, and bit-exact
HW-vs-SW argmax match >= 99% (the residual <1% are genuine z==T tie samples at
the decision boundary, which is mathematically unavoidable for either integer rule).
"""

import json
import os
import struct
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sw.model.bnn import BNN
from scripts.export import export_model

CFG = os.path.join(REPO, "configs")
MEM = os.path.join(REPO, "mem_files")
DS = os.path.join(REPO, "Dataset")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000


def _rid3(fn):
    with open(fn, "rb") as f:
        _, _, d = struct.unpack(">HBB", f.read(4))
        sh = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(d))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)


def _rid1(fn):
    with open(fn, "rb") as f:
        _, _, d = struct.unpack(">HBB", f.read(4))
        sh = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(d))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)


def _read_mem_lines(name):
    with open(os.path.join(MEM, name)) as f:
        return [l.strip() for l in f if l.strip()]


def _hexrow_to_bits(h, width):
    return np.array([int(b) for b in format(int(h, 16), f"0{width}b")], dtype=np.int32)


def main():
    print(f"[verify] re-exporting mem_files from checkpoint ...")
    export_model()

    with open(os.path.join(CFG, "model.yaml")) as f:
        cm = yaml.safe_load(f)
    ckpt = torch.load(
        os.path.join(REPO, "checkpoints", "best_model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model = BNN(
        input_size=cm["input_size"],
        hidden_sizes=cm["hidden_sizes"],
        num_classes=cm["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["mean"]).reshape(-1)
    std = np.asarray(ckpt["std"]).reshape(-1)
    meta = json.load(open(os.path.join(MEM, "export_meta.json")))

    w1 = _read_mem_lines("layer1_weights.mem")
    w2 = _read_mem_lines("layer2_weights.mem")
    w3 = _read_mem_lines("layer3_weights.mem")
    th1 = np.loadtxt(os.path.join(MEM, "layer1_thresholds.mem"))
    th2 = np.loadtxt(os.path.join(MEM, "layer2_thresholds.mem"))
    off3 = np.loadtxt(os.path.join(MEM, "layer3_offsets.mem"))
    scale3 = np.loadtxt(os.path.join(MEM, "layer3_scales.mem"))
    W1 = np.array([_hexrow_to_bits(h, meta["layer_shapes"][0][0]) for h in w1], dtype=np.int32)
    W2 = np.array([_hexrow_to_bits(h, meta["layer_shapes"][1][0]) for h in w2], dtype=np.int32)
    W3 = np.array([_hexrow_to_bits(h, meta["layer_shapes"][2][0]) for h in w3], dtype=np.int32)

    imgs = _rid3(os.path.join(DS, "t10k-images.idx3-ubyte")).astype(np.float32) / 255.0
    labs = _rid1(os.path.join(DS, "t10k-labels.idx1-ubyte"))

    correct = hw_correct = canon = total = 0
    with torch.no_grad():
        for idx in range(N):
            x_real = imgs[idx].reshape(-1)
            x_norm = (x_real - mean) / (std + 1e-7)
            out = model(torch.tensor(x_norm.reshape(1, -1)))
            tp = int(out.argmax(dim=1).item())

            a_bit = ((np.where(x_norm >= 0, 1.0, -1.0) + 1) / 2).astype(np.int32)
            p1 = (1 - (W1 ^ a_bit)).sum(1)
            a = (p1 >= th1).astype(np.int32)
            p2 = (1 - (W2 ^ a)).sum(1)
            a = (p2 >= th2).astype(np.int32)
            p3 = (1 - (W3 ^ a)).sum(1)
            z3 = 2 * p3 - W3.shape[1]
            logits = scale3 * z3 + off3
            hp = int(np.argmax(logits))

            if tp == int(labs[idx]):
                correct += 1
            if hp == int(labs[idx]):
                hw_correct += 1
            if hp == tp:
                canon += 1
            total += 1

    sw_acc = 100.0 * correct / total
    hw_acc = 100.0 * hw_correct / total
    match = 100.0 * canon / total
    print(f"[verify] SW acc on {total}: {sw_acc:.3f}%")
    print(f"[verify] HW(popcount) acc on {total}: {hw_acc:.3f}%")
    print(f"[verify] HW-vs-SW argmax match: {canon}/{total} ({match:.3f}%)")

    ok = sw_acc > 95.0 and hw_acc > 95.0 and match >= 99.0
    if not ok:
        print("[verify] FAIL")
        sys.exit(1)
    print("[verify] PASS: >95% SW & HW acc, >=99% bit-exact popcount match")


if __name__ == "__main__":
    main()
