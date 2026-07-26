import os
import torch
import numpy as np
from pathlib import Path
from config import get_config
from sw.model import BNN

def export_weights():
    cfg = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BNN(
        input_size=cfg["model"]["input_size"],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No trained model found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    mem_dir = Path(cfg["mem_dir"])
    mem_dir.mkdir(parents=True, exist_ok=True)

    # Process layers and export
    # The network is: BinarizeLinear -> BatchNorm -> Hardtanh -> ...
    # We will export:
    # 1. Binarized Weights (sign) packed into hexadecimal format.
    # 2. Thresholds for activations.
    # Since BatchNorm modifies the threshold, we combine them:
    # y = BatchNorm(w*x) = gamma * (w*x - mu)/sqrt(var + eps) + beta
    # we want y > 0 to be a '1' output, and y <= 0 to be '-1' output, since Hardtanh/Sign activation follows.
    # gamma * (w*x - mu) / sigma + beta > 0
    # gamma/sigma * w*x > gamma*mu/sigma - beta
    # w*x > mu - beta*sigma/gamma (if gamma > 0)
    # Threshold = mu - beta*sigma/gamma

    # In chunked popcount BNNs, inputs are -1 and +1.
    # dot product ranges from -N to N.
    # Let's extract weights and thresholds directly.
    # For a simple export, we can just save the raw binarized weights and combined thresholds.

    linear_idx = 1

    with torch.no_grad():
        features = list(model.features)
        for i in range(len(features)):
            if isinstance(features[i], torch.nn.Linear):
                # Binarized weights: sign(w), convert to 0 and 1
                # Where 1 represents +1 and 0 represents -1
                weights = features[i].weight.data
                bin_weights = torch.sign(weights).add_(weights == 0).float()
                # Map -1 -> 0, +1 -> 1
                bin_weights_01 = ((bin_weights + 1) / 2).int().cpu().numpy()

                # Check for BatchNorm layer immediately after
                has_bn = i + 1 < len(features) and isinstance(features[i+1], torch.nn.BatchNorm1d)

                if has_bn:
                    bn = features[i+1]
                    gamma = bn.weight.data
                    beta = bn.bias.data
                    mu = bn.running_mean.data
                    var = bn.running_var.data
                    eps = bn.eps

                    sigma = torch.sqrt(var + eps)
                    # Threshold T = mu - beta * sigma / gamma
                    thresholds = mu - (beta * sigma / gamma)

                    # If gamma is negative, the inequality flips. For now, assume gamma > 0 or handle it:
                    # Let's just save threshold. If gamma < 0, we flip the threshold logic in hardware (or invert weights)
                    # We will save thresholds as float for now, or int since w*x is integer.
                    # Since w*x step is 2, rounding to nearest int is fine.
                    thresholds = thresholds.cpu().numpy()
                else:
                    thresholds = np.zeros(bin_weights_01.shape[0])

                # Save to mem file
                # Write weights in hex. One row per output neuron.
                # Since weights can be large (e.g. 784 bits), we can format them as large hex strings.
                w_filepath = mem_dir / f"layer{linear_idx}_weights.mem"
                t_filepath = mem_dir / f"layer{linear_idx}_thresholds.mem"

                with open(w_filepath, "w") as fw:
                    for row in bin_weights_01:
                        # Pack row into an integer
                        # Little endian or big endian?
                        # Let's pack as a big integer and convert to hex
                        # Row length e.g. 784
                        bit_str = "".join(str(b) for b in row)
                        # Hex string, padding to correct number of hex digits
                        hex_len = (len(bit_str) + 3) // 4
                        hex_val = hex(int(bit_str, 2))[2:].zfill(hex_len)
                        fw.write(f"{hex_val}\n")

                with open(t_filepath, "w") as ft:
                    for t in thresholds:
                        # Writing as integer since w*x is integer (or floats if preferred)
                        # We'll write as float strings
                        ft.write(f"{t:.4f}\n")

                print(f"Exported Layer {linear_idx}: shape {bin_weights_01.shape}")
                linear_idx += 1

    print("Export complete.")

if __name__ == "__main__":
    export_weights()
