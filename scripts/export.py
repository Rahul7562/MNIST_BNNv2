"""Export a trained BNN to the hardware contract files in mem_files/.

Source of truth: docs/ARCHITECTURE.md §5 (BatchNorm folding) and §6 (export format).

For each layer l with binarized weight W_l in {-1,+1}^{M x N} and BatchNorm
(gamma, beta, mu, sigma):
  * Enforce gamma > 0 by folding sign(gamma) into the weight row:
        w_new = w * sign(gamma)          (in {-1, +1})
        T     = sign(gamma)*mu - beta*sigma/|gamma|
  * Hidden layer: HW computes P = popcount(XNOR(w_new, a)), z = 2*P - N, then
    decides a' = +1 iff z > T  <=>  P >= floor((T + N)/2) + 1  (integer compare).
  * Output layer (no activation): HW computes z_c = 2*P_c - N (true dot), then
        logit_c = (gamma_c/sigma_c)*z_c + (beta_c - gamma_c*mu_c/sigma_c)
    i.e. per-class scale = gamma_c/sigma_c and offset = beta_c - gamma_c*mu_c/sigma_c.
HW reads ONLY these .mem files + export_meta.json.
"""

import json
import os

import numpy as np
import torch
import yaml
from pathlib import Path

from config import get_config
from sw.model import BNN, binarize_weight


def _bits_to_hex(bits):
    """bits: iterable of 0/1 (MSB-first). Return big-endian hex string (no 0x)."""
    bit_str = "".join(str(int(b)) for b in bits)
    hex_len = (len(bit_str) + 3) // 4
    return format(int(bit_str, 2), f"0{hex_len}x")


def export_model():
    cfg = get_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    mem_dir = Path(cfg["mem_dir"])
    mem_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(train_cfg["checkpoint_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BNN(
        input_size=model_cfg["input_size"],
        hidden_sizes=model_cfg["hidden_sizes"],
        num_classes=model_cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = np.asarray(ckpt["mean"]).reshape(-1)
    std = np.asarray(ckpt["std"]).reshape(-1)

    meta = {
        "input_dims": [1, 28, 28],
        "layer_shapes": [],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "binarize_method": "sign_ste",
        "num_classes": model_cfg["num_classes"],
        "popcount_w_per_neuron": [],
        "q_format": "binary_1_0_mapped_from_plus1_minus1",
    }

    features = list(model.features)
    layer_idx = 1
    i = 0
    with torch.no_grad():
        while i < len(features):
            lin = features[i]
            if not isinstance(lin, torch.nn.Linear):
                i += 1
                continue
            bn = features[i + 1] if (i + 1 < len(features)) else None

            # Binarized weights in {-1,+1}, stored as bit (1 = +1).
            w = binarize_weight(lin.weight).detach().numpy() * 2.0 - 1.0  # {-1,+1}
            w_01 = (w > 0).astype(np.int32)  # 1 = +1

            is_output = bn is None or (i + 2 >= len(features))

            if bn is not None:
                gamma = bn.weight.detach().numpy()
                beta = bn.bias.detach().numpy()
                mu = bn.running_mean.detach().numpy()
                var = bn.running_var.detach().numpy()
                sigma = np.sqrt(var + bn.eps)

                # Enforce gamma > 0 by folding sign(gamma) into the weight row.
                sgn = np.sign(gamma)
                sgn[sgn == 0] = 1.0
                w = w * sgn.reshape(-1, 1)
                w_01 = (w > 0).astype(np.int32)
                T = sgn * mu - (beta * sigma) / np.abs(gamma)
            else:
                T = np.zeros(w.shape[0])

            N = w.shape[1]  # layer input width (popcount basis)
            popcount_w = w_01.sum(axis=1)
            meta["popcount_w_per_neuron"].append(popcount_w.tolist())

            # Weights .mem (big-endian hex, one line per output neuron).
            with open(mem_dir / f"layer{layer_idx}_weights.mem", "w") as f:
                for row in w_01:
                    f.write(_bits_to_hex(row) + "\n")

            if not is_output:
                # Hidden layer: HW computes P = popcount(XNOR(w_bit, a_bit)),
                # z = 2*P - N, decision a' = +1 iff z > T iff P > (T + N)/2.
                # Since P is integer: P >= floor((T+N)/2) + 1.
                Th = np.floor((T + N) / 2.0).astype(np.int64) + 1
                with open(mem_dir / f"layer{layer_idx}_thresholds.mem", "w") as f:
                    for th in Th:
                        f.write(f"{int(th)}\n")
            else:
                # Output layer: HW computes z_c = 2*P_c - N (true dot), then
                # logit_c = (gamma_c/sigma_c)*z_c + (beta_c - gamma_c*mu_c/sigma_c).
                # The scale gamma_c/sigma_c is NOT 1 in general, so it must be
                # stored per class (a float multiply), not folded into an offset.
                scale_c = gamma / sigma
                off_c = beta - gamma * mu / sigma
                with open(mem_dir / f"layer{layer_idx}_scales.mem", "w") as f:
                    for s in scale_c:
                        f.write(f"{float(s)}\n")
                with open(mem_dir / f"layer{layer_idx}_offsets.mem", "w") as f:
                    for off in off_c:
                        f.write(f"{float(off)}\n")

            meta["layer_shapes"].append([w.shape[1], w.shape[0]])
            print(f"Exported layer {layer_idx}: weights {w_01.shape}")
            layer_idx += 1
            i += 3 if bn is not None else 1

    with open(mem_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Export complete -> {mem_dir}")


if __name__ == "__main__":
    export_model()
