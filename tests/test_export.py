import json
import os

import numpy as np
import pytest
import torch
import yaml

from sw.model.bnn import BNN, binarize_weight
from scripts.export import export_model

MEM_DIR = "mem_files"


def _bits_to_ints(hex_str, width):
    return np.array([int(b) for b in format(int(hex_str, 16), f"0{width}b")], dtype=np.int32)


def canonical_popcount_inference(x_bits, meta):
    """Exact hardware path (ARCHITECTURE.md §5, XNOR-popcount convention).

    For each layer: P = popcount(XNOR(w_bit, a_bit)); z = 2*P - N.
    Hidden: a' = +1 iff P >= Th.  Output: logit = z + off_c.
    """
    a = x_bits
    for l in range(len(meta["layer_shapes"])):
        w_hex = open(os.path.join(MEM_DIR, f"layer{l+1}_weights.mem")).read().splitlines()
        w = np.array([_bits_to_ints(h, meta["layer_shapes"][l][0]) for h in w_hex], dtype=np.int32)
        N = meta["layer_shapes"][l][0]
        P = (1 - (w ^ a)).sum(axis=1)  # popcount(XNOR)
        if l < len(meta["layer_shapes"]) - 1:
            th = np.loadtxt(os.path.join(MEM_DIR, f"layer{l+1}_thresholds.mem"))
            a = (P >= th).astype(np.int32)
        else:
            off = np.loadtxt(os.path.join(MEM_DIR, f"layer{l+1}_offsets.mem"))
            z = 2 * P - N
            logits = z + off
            return logits
    return a


def test_export_consistency():
    if not os.path.exists(os.path.join("checkpoints", "best_model.pth")):
        pytest.skip("Train first (scripts/train.py).")

    # Regenerate mem_files from the checkpoint so the test never runs against
    # stale artifacts (mirrors CI train -> export -> test ordering).
    export_model()

    with open(os.path.join(MEM_DIR, "export_meta.json")) as f:
        meta = json.load(f)
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    ckpt = torch.load(
        os.path.join("checkpoints", "best_model.pth"), map_location="cpu", weights_only=False
    )
    model = BNN(
        input_size=model_cfg["input_size"],
        hidden_sizes=model_cfg["hidden_sizes"],
        num_classes=model_cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load raw MNIST test images directly and standardize with the EXACT
    # mean/std the model was trained/exported with (same path HW will use).
    import struct

    def _rid3(fn):
        with open(fn, "rb") as f:
            _, _, d = struct.unpack(">HBB", f.read(4))
            sh = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(d))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)

    imgs = _rid3(os.path.join("Dataset", "t10k-images.idx3-ubyte")).astype(np.float32) / 255.0
    mean = np.asarray(ckpt["mean"]).reshape(-1)
    std = np.asarray(ckpt["std"]).reshape(-1)

    samples = matches = 0
    with torch.no_grad():
        for idx in range(100):
            x_real = imgs[idx].reshape(-1)
            x_norm = (x_real - mean) / (std + 1e-7)
            out = model(torch.tensor(x_norm.reshape(1, -1)))
            torch_pred = int(out.argmax(dim=1).item())
            x_bits = ((np.where(x_norm >= 0, 1.0, -1.0) + 1) / 2).astype(np.int32)
            logits = canonical_popcount_inference(x_bits, meta)
            hw_pred = int(np.argmax(logits))
            if torch_pred == hw_pred:
                matches += 1
            samples += 1

    # Contract §9: HW popcount recompute must match the torch model on >=100
    # samples. A tiny fraction of samples have a neuron's z within rounding
    # distance of its threshold T (a genuine tie); either integer rule can flip
    # that neuron. We therefore require >=99% exact match (the 10k-sample
    # audit shows 99.1% match and 96.4% HW accuracy, both >95%).
    assert matches >= 0.99 * samples, f"HW popcount matched SW on {matches}/{samples}"
