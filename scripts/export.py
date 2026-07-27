import os
import yaml
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from sw.model.bnn import BNN, binary_sign, binary_weight

def export_model(model_path, config_model_path, config_train_path):
    with open(config_model_path, 'r') as f:
        config_model = yaml.safe_load(f)['model']

    with open(config_train_path, 'r') as f:
        config_train = yaml.safe_load(f)['training']

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = BNN(
        input_size=config_model['input_size'],
        hidden_sizes=config_model['hidden_sizes'],
        num_classes=config_model['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    out_dir = Path(config_train['mem_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        'input_dims': [1, 28, 28],
        'layer_shapes': [],
        'mean': float(checkpoint['mean']),
        'std': float(checkpoint['std']),
        'binarize_method': 'sign_ste',
        'num_classes': config_model['num_classes'],
        'popcount_w_per_neuron': [],
        'q_format': 'binary_1_0_mapped_from_plus1_minus1'
    }

    layer_idx = 1
    i = 0
    while i < len(model.features):
        linear = model.features[i]
        bn = model.features[i+1]

        # w is in {0, 1} conceptually
        w = binary_weight(linear.weight).detach().numpy()
        w_bin = w.astype(np.int32)

        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        mu = bn.running_mean.detach().numpy()
        var = bn.running_var.detach().numpy()
        eps = bn.eps
        sigma = np.sqrt(var + eps)

        gamma_abs = np.abs(gamma)

        popcount_w = w_bin.sum(axis=1)
        meta['popcount_w_per_neuron'].append(popcount_w.tolist())

        weights_file = out_dir / f"layer{layer_idx}_weights.mem"
        with open(weights_file, 'w') as f:
            for row in w_bin:
                row_str = "".join(str(b) for b in row)
                hex_str = hex(int(row_str, 2))[2:].zfill((len(row) + 3) // 4)
                f.write(f"{hex_str}\n")

        is_output = (layer_idx == len(config_model['hidden_sizes']) + 1)

        if not is_output:
            T = mu - beta * sigma / gamma_abs
            Th = np.floor((T + popcount_w) / 2.0) + 1.0

            thresh_file = out_dir / f"layer{layer_idx}_thresholds.mem"
            with open(thresh_file, 'w') as f:
                for th in Th:
                    f.write(f"{int(th)}\n")
        else:
            off_c = beta - gamma * mu / sigma

            offsets_file = out_dir / f"layer{layer_idx}_offsets.mem"
            with open(offsets_file, 'w') as f:
                for off in off_c:
                    f.write(f"{float(off)}\n")

        meta['layer_shapes'].append([w.shape[1], w.shape[0]])

        i += 3 if not is_output else 2
        layer_idx += 1

    with open(out_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Exported all artifacts to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--config_model', type=str, default='configs/model.yaml')
    parser.add_argument('--config_train', type=str, default='configs/training.yaml')
    args = parser.parse_args()

    export_model(args.model_path, args.config_model, args.config_train)

if __name__ == '__main__':
    main()
