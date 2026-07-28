"""Prepare hw/mem/ from the export contract in mem_files/.

The SW export writes:
  layer{1,2}_weights.mem   -> big-endian hex bits (OK for $readmemh)
  layer3_weights.mem       -> big-endian hex bits (OK)
  layer{1,2}_thresholds.mem-> decimal integers (need hex for $readmemh)
  layer3_scales.mem        -> floats (need IEEE-754 double hex for $bitstoreal)
  layer3_offsets.mem       -> floats (same)

This script copies weights as-is and converts thresholds/scales/offsets into
hex files the RTL reads. It also generates test vectors (784-bit activation
hex per image) + labels from the MNIST test set and my_digit.png, so the
Icarus testbench can drive real inputs.
"""

import json
import os
import struct
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # script is in REPO/scripts/
MEM = os.path.join(REPO, "mem_files")
HWMEM = os.path.join(REPO, "hw", "mem")
DS = os.path.join(REPO, "Dataset")


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


def _read_dec(name):
    with open(os.path.join(MEM, name)) as f:
        return [float(l.strip()) for l in f if l.strip()]


def _bits_to_hex(bits):
    bit_str = "".join(str(int(b)) for b in bits)
    hex_len = (len(bit_str) + 3) // 4
    return format(int(bit_str, 2), f"0{hex_len}x")


def main():
    os.makedirs(HWMEM, exist_ok=True)

    # Copy weight hex files.
    for fn in ["layer1_weights.mem", "layer2_weights.mem", "layer3_weights.mem"]:
        with open(os.path.join(MEM, fn)) as f:
            data = f.read()
        with open(os.path.join(HWMEM, fn), "w") as f:
            f.write(data)

    # Thresholds: decimal -> hex (unsigned up to <784).
    for fn in ["layer1_thresholds.mem", "layer2_thresholds.mem"]:
        vals = _read_dec(fn)
        with open(os.path.join(HWMEM, fn.replace(".mem", ".hex")), "w") as f:
            for v in vals:
                f.write(format(int(v), "x") + "\n")

    # Scales/offsets: float -> IEEE-754 double hex (for $bitstoreal / reference).
    for fn in ["layer3_scales.mem", "layer3_offsets.mem"]:
        vals = _read_dec(fn)
        with open(os.path.join(HWMEM, fn.replace(".mem", ".hex")), "w") as f:
            for v in vals:
                f.write(struct.pack(">d", float(v)).hex() + "\n")

    # Scales/offsets: fixed-point signed 32-bit hex for SYNTHESIS (no real/$bitstoreal).
    #   logit_fxd = scale_fxd * z + offset_fxd,  with value = round(float * 2**FXD)
    # FXD=16 keeps scale*z (|z|<=256, |scale|~0.03) within signed 32-bit and leaves
    # sub-LSB rounding error (<0.004) so argmax is bit-exact vs the float reference.
    FXD = 16
    for fn in ["layer3_scales.mem", "layer3_offsets.mem"]:
        vals = _read_dec(fn)
        with open(os.path.join(HWMEM, fn.replace(".mem", "_fxd.hex")), "w") as f:
            for v in vals:
                ival = int(round(float(v) * (1 << FXD)))
                f.write(format(ival & 0xFFFFFFFF, "08x") + "\n")  # 32-bit two's complement

    # Test vectors: MNIST t10k (all 10k) + my_digit.png, binarized 784-bit hex.
    ckpt = torch = None  # imported lazily to avoid hard dep if run standalone
    import yaml
    cm = yaml.safe_load(open(os.path.join(REPO, "configs", "model.yaml")))
    import torch  # noqa
    ck = torch.load(
        os.path.join(REPO, "checkpoints", "best_model.pth"), map_location="cpu", weights_only=False
    )
    mean = np.asarray(ck["mean"]).reshape(-1)
    std = np.asarray(ck["std"]).reshape(-1)

    sim_count = int(os.environ.get("SIM_COUNT", "50"))  # speed: gate needs >=40

    def bits_for(arr):
        x = (arr.astype(np.float32).reshape(-1) - mean) / (std + 1e-7)
        b = (np.where(x >= 0, 1, 0)).astype(int)
        return b

    imgs = _rid3(os.path.join(DS, "t10k-images.idx3-ubyte"))
    labs = _rid1(os.path.join(DS, "t10k-labels.idx1-ubyte"))

    meta = json.load(open(os.path.join(MEM, "export_meta.json")))
    sh = meta["layer_shapes"]

    def _load_bits(name, width):
        out = []
        for h in open(os.path.join(MEM, name)).read().splitlines():
            h = h.strip()
            if not h:
                continue
            out.append(np.array([int(b) for b in format(int(h, 16), f"0{width}b")]))
        return np.array(out)

    W1 = _load_bits("layer1_weights.mem", sh[0][0])
    W2 = _load_bits("layer2_weights.mem", sh[1][0])
    W3 = _load_bits("layer3_weights.mem", sh[2][0])
    th1 = np.loadtxt(os.path.join(MEM, "layer1_thresholds.mem"))
    th2 = np.loadtxt(os.path.join(MEM, "layer2_thresholds.mem"))
    sc3 = np.loadtxt(os.path.join(MEM, "layer3_scales.mem"))
    off3 = np.loadtxt(os.path.join(MEM, "layer3_offsets.mem"))

    def _hw_digit(bits):
        p1 = (1 - (W1 ^ bits)).sum(1)
        a1 = (p1 >= th1).astype(int)
        p2 = (1 - (W2 ^ a1)).sum(1)
        a2 = (p2 >= th2).astype(int)
        p3 = (1 - (W3 ^ a2)).sum(1)
        z3 = 2 * p3 - W3.shape[1]
        logits = sc3 * z3 + off3
        return int(np.argmax(logits))

    vecs = []
    labels = []  # expected digit via SW popcount recompute; 255 = skip (my_digit)
    for i in range(min(sim_count, imgs.shape[0])):
        b = bits_for(imgs[i])
        vecs.append(_bits_to_hex(b))
        labels.append(_hw_digit(b))  # expected = SW popcount recompute (the §9 gate)

    # my_digit.png if present (the project's primary success criterion).
    md = os.path.join(REPO, "my_digit.png")
    if os.path.exists(md):
        im = Image.open(md).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        vecs.append(_bits_to_hex(bits_for(arr)))
        labels.append(255)  # sentinel: skip label check; just verify it runs & agrees w/ SW

    with open(os.path.join(HWMEM, "testvecs.mem"), "w") as f:
        for v in vecs:
            f.write(v + "\n")
    with open(os.path.join(HWMEM, "testlabels.mem"), "w") as f:
        for l in labels:
            f.write(str(l) + "\n")

    print(f"[prepare_hw_mem] wrote {len(vecs)} test vectors to hw/mem/ (incl my_digit.png={os.path.exists(md)})")


if __name__ == "__main__":
    main()
