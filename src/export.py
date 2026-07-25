import torch
import numpy as np
import os
import json
from train import BNN, binarize

def export_weights_mem(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def to_bin_str(tensor, invert=False):
        arr = tensor.detach().cpu().numpy().flatten()
        if invert:
            arr = (arr > 0).astype(int)
        else:
            arr = np.where(arr > 0, 1, 0)
        return "".join(map(str, arr))

    def write_mem(filename, bin_strs):
        with open(os.path.join(out_dir, filename), "w") as f:
            for s in bin_strs:
                hex_val = hex(int(s, 2))[2:]
                expected_len = (len(s) + 3) // 4
                f.write(hex_val.zfill(expected_len) + "\n")

    print("Exporting Conv1 weights...")
    w1 = binarize(model.conv1.weight)
    w1_strs = [to_bin_str(w1[i]) for i in range(w1.size(0))]
    write_mem("conv1_weights.mem", w1_strs)

    print("Exporting Conv2 weights...")
    w2 = binarize(model.conv2.weight)
    w2_strs = [to_bin_str(w2[i]) for i in range(w2.size(0))]
    write_mem("conv2_weights.mem", w2_strs)

    print("Exporting Conv3 weights...")
    w3 = binarize(model.conv3.weight)
    w3_strs = [to_bin_str(w3[i]) for i in range(w3.size(0))]
    write_mem("conv3_weights.mem", w3_strs)

    print("Exporting FC1 weights...")
    w4 = binarize(model.fc1.weight)
    w4_strs = [to_bin_str(w4[i]) for i in range(w4.size(0))]
    write_mem("fc1_weights.mem", w4_strs)

    print("Exporting FC2 weights...")
    w5 = binarize(model.fc2.weight)
    w5_strs = [to_bin_str(w5[i]) for i in range(w5.size(0))]
    write_mem("fc2_weights.mem", w5_strs)

    # Save .npy format
    np.save(os.path.join(out_dir, "conv1_weights.npy"), w1.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "conv2_weights.npy"), w2.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "conv3_weights.npy"), w3.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "fc1_weights.npy"), w4.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "fc2_weights.npy"), w5.detach().cpu().numpy())

    # Save .json format (we can just dump shapes or small arrays, but let's dump full for inspection as requested)
    weights_dict = {
        "conv1": w1.detach().cpu().numpy().tolist(),
        "conv2": w2.detach().cpu().numpy().tolist(),
        "conv3": w3.detach().cpu().numpy().tolist(),
        "fc1": w4.detach().cpu().numpy().tolist(),
        "fc2": w5.detach().cpu().numpy().tolist()
    }
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights_dict, f)

    def export_bn_thresholds(bn, N, filename):
        gamma = bn.weight.detach().cpu().numpy()
        beta = bn.bias.detach().cpu().numpy()
        mean = bn.running_mean.detach().cpu().numpy()
        var = bn.running_var.detach().cpu().numpy()
        eps = bn.eps

        std = np.sqrt(var + eps)

        thresholds = []
        inverts = []
        for i in range(len(gamma)):
            thresh_real = (N + mean[i] - (beta[i] * std[i] / gamma[i])) / 2.0
            if gamma[i] > 0:
                thresh_int = int(np.floor(thresh_real) + 1)
                invert = 0
            else:
                thresh_int = int(np.ceil(thresh_real) - 1)
                invert = 1

            thresh_int = max(0, min(thresh_int, N))
            thresholds.append(thresh_int)
            inverts.append(invert)

        with open(os.path.join(out_dir, filename), "w") as f:
            for t, inv in zip(thresholds, inverts):
                val = (inv << 16) | (t & 0xFFFF)
                f.write(f"{val:05x}\n")

        # save npy and json for thresholds too
        np.save(os.path.join(out_dir, filename.replace(".mem", ".npy")), np.array(thresholds))
        return thresholds, inverts

    print("Exporting BN thresholds...")
    N1 = 1 * 3 * 3
    N2 = 32 * 3 * 3
    N3 = 64 * 3 * 3
    N4 = 64 * 3 * 3
    N5 = 64

    t1, i1 = export_bn_thresholds(model.bn1, N1, "bn1_thresh.mem")
    t2, i2 = export_bn_thresholds(model.bn2, N2, "bn2_thresh.mem")
    t3, i3 = export_bn_thresholds(model.bn3, N3, "bn3_thresh.mem")
    t4, i4 = export_bn_thresholds(model.bn4, N4, "bn4_thresh.mem")
    t5, i5 = export_bn_thresholds(model.bn5, N5, "bn5_thresh.mem")

    thresh_dict = {
        "bn1": {"thresholds": t1, "inverts": i1},
        "bn2": {"thresholds": t2, "inverts": i2},
        "bn3": {"thresholds": t3, "inverts": i3},
        "bn4": {"thresholds": t4, "inverts": i4},
        "bn5": {"thresholds": t5, "inverts": i5},
    }
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(thresh_dict, f, indent=4)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = BNN().to(device)
    if os.path.exists('models/bnn_mnist.pth'):
        model.load_state_dict(torch.load('models/bnn_mnist.pth', map_location=device))
        print("Loaded trained model for export.")
    else:
        print("No trained model found! Exporting random initialized weights.")
    export_weights_mem(model, "mem_files")
