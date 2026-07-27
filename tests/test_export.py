import os
import json
import torch
import numpy as np
import pytest

from sw.model.bnn import BNN
from sw.training.dataset import get_dataloaders

def popcount_inference(x_bin, mem_dir, meta):
    # x_bin in {0, 1} with shape (784,)
    a = x_bin

    # Layer 1
    w1_hex = open(os.path.join(mem_dir, "layer1_weights.mem")).read().splitlines()
    th1 = np.loadtxt(os.path.join(mem_dir, "layer1_thresholds.mem"))

    a_next = np.zeros(len(w1_hex), dtype=np.int32)
    for i, w_hex in enumerate(w1_hex):
        w_bin = np.array([int(b) for b in bin(int(w_hex, 16))[2:].zfill(meta['layer_shapes'][0][0])])
        pop = np.sum(w_bin & a)
        a_next[i] = 1 if pop >= th1[i] else 0

    a = a_next

    # Layer 2
    w2_hex = open(os.path.join(mem_dir, "layer2_weights.mem")).read().splitlines()
    th2 = np.loadtxt(os.path.join(mem_dir, "layer2_thresholds.mem"))

    a_next = np.zeros(len(w2_hex), dtype=np.int32)
    for i, w_hex in enumerate(w2_hex):
        w_bin = np.array([int(b) for b in bin(int(w_hex, 16))[2:].zfill(meta['layer_shapes'][1][0])])
        pop = np.sum(w_bin & a)
        a_next[i] = 1 if pop >= th2[i] else 0

    a = a_next

    # Layer 3
    w3_hex = open(os.path.join(mem_dir, "layer3_weights.mem")).read().splitlines()
    off3 = np.loadtxt(os.path.join(mem_dir, "layer3_offsets.mem"))

    logits = np.zeros(len(w3_hex), dtype=np.float32)
    for i, w_hex in enumerate(w3_hex):
        w_bin = np.array([int(b) for b in bin(int(w_hex, 16))[2:].zfill(meta['layer_shapes'][2][0])])
        pop = np.sum(w_bin & a)
        z = 2 * pop - np.sum(w_bin)
        logits[i] = z + off3[i]

    return logits

def test_export_consistency():
    if not os.path.exists('mem_files/export_meta.json'):
        pytest.skip("Export artifacts not found. Run export first.")

    with open('mem_files/export_meta.json', 'r') as f:
        meta = json.load(f)

    _, test_loader, _, _ = get_dataloaders('Dataset', batch_size=1)

    import yaml
    with open('configs/model.yaml', 'r') as f:
        config_model = yaml.safe_load(f)['model']

    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)
    model = BNN(
        input_size=config_model['input_size'],
        hidden_sizes=config_model['hidden_sizes'],
        num_classes=config_model['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    samples = 0
    matches = 0

    bn_out = model.features[7]
    gamma = bn_out.weight.detach().numpy()
    beta = bn_out.bias.detach().numpy()
    mu = bn_out.running_mean.detach().numpy()
    sigma = np.sqrt(bn_out.running_var.detach().numpy() + bn_out.eps)

    with torch.no_grad():
        for data, target in test_loader:
            if samples >= 100:
                break

            out = model(data)
            torch_pred = out.argmax(dim=1).item()

            x_bin = ((data[0].numpy() + 1) / 2).astype(np.int32)

            logits_hw = popcount_inference(x_bin, 'mem_files', meta)

            # Now we compute exact match
            # y_torch = (gamma / sigma) * z + beta - gamma * mu / sigma
            # logits_hw = z + beta - gamma * mu / sigma
            # so exact:
            logits_exact = (np.abs(gamma) / sigma) * (logits_hw - (beta - gamma * mu / sigma)) + (beta - gamma * mu / sigma)

            hw_pred = np.argmax(logits_exact)

            if torch_pred == hw_pred:
                matches += 1

            samples += 1

    assert matches == 100, f"Popcount recompute matched torch argmax on {matches}/100 samples."
