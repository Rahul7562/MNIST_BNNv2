import subprocess
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add root directory to python path for imports
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from config import get_config
from sw.model.bnn import BNN
from sw import dataset

def main():
    cfg = get_config()

    device = torch.device("cpu")
    model = BNN(
        input_size=cfg["model"]["input_size"],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    checkpoint_path = Path(cfg["training"]["checkpoint_dir"]) / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset_dir = Path(cfg["dataset_dir"])
    test_images_path = dataset_dir / "t10k-images.idx3-ubyte"
    test_labels_path = dataset_dir / "t10k-labels.idx1-ubyte"

    test_images = dataset.read_idx3_images(test_images_path)
    test_labels = dataset.read_idx1_labels(test_labels_path)

    # Calculate HW equivalent scores
    gamma = model.features[8].weight.data
    beta = model.features[8].bias.data
    mu = model.features[8].running_mean.data
    var = model.features[8].running_var.data
    eps = model.features[8].eps
    sigma = torch.sqrt(var + eps)
    thresh_l3 = mu - beta * sigma / gamma

    num_tests = 40
    correct = 0
    hw_dir = root_dir / "hw"

    print(f"Starting verification of {num_tests} images...")

    subprocess.run(["make", "compile"], cwd=str(hw_dir), check=True)

    for i in range(num_tests):
        img = test_images[i]
        label = test_labels[i]

        sw_input_tensor = torch.tensor([1.0 if p >= 127 else -1.0 for p in img.flatten()], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Get SW output exactly as HW computes it (HW only uses thresholds, ignoring gamma variance)
            # In Phase 1 we accepted that exporting thresholds alone is an approximation for L3 argmax.
            # To ensure the HW is doing exactly what it was exported to do, we compare it against the threshold-based score.
            x = sw_input_tensor.view(1, -1)
            x1 = model.features[1](x)
            x1_bin = torch.sign(model.features[3](model.features[2](x1))).add_(model.features[3](model.features[2](x1)) == 0).float()
            x2 = model.features[4](x1_bin)
            x2_bin = torch.sign(model.features[6](model.features[5](x2))).add_(model.features[6](model.features[5](x2)) == 0).float()
            x3 = model.features[7](x2_bin)

            hw_equivalent_score = x3 - thresh_l3
            hw_eq_pred = hw_equivalent_score.argmax().item()

        bits = "".join("1" if p >= 127 else "0" for p in img.flatten())
        with open(hw_dir.parent / "mem_files" / "input.mem", "w") as f:
            f.write(bits + "\n")

        result = subprocess.run(["vvp", "bnn_sim"], cwd=str(hw_dir), capture_output=True, text=True)
        hw_pred_str = [line for line in result.stdout.split('\n') if 'Predicted:' in line]
        if not hw_pred_str:
            print(f"Test {i}: HW simulation failed to output prediction")
            continue

        hw_pred = int(hw_pred_str[0].split(',')[0].split(':')[1].strip())

        match = (hw_eq_pred == hw_pred)
        if match:
            correct += 1
            print(f"Test {i} [{label}]: PASS (SW/HW equivalent expected: {hw_eq_pred}, HW got: {hw_pred})")
        else:
            print(f"Test {i} [{label}]: FAIL (SW/HW equivalent expected: {hw_eq_pred}, HW got: {hw_pred})")

    print("-" * 30)
    print(f"Verification Results: {correct}/{num_tests} correct ({(correct/num_tests)*100:.1f}%)")
    if correct != num_tests:
        sys.exit(1)

    # -------------------------------------------------------------
    # Now test against my_digit.png
    # -------------------------------------------------------------
    print("\nVerifying my_digit.png...")
    subprocess.run(["python3", "hw/scripts/convert_image.py", "my_digit.png", "input.mem"], cwd=str(root_dir), check=True)

    result = subprocess.run(["vvp", "bnn_sim"], cwd=str(hw_dir), capture_output=True, text=True)
    hw_pred_str = [line for line in result.stdout.split('\n') if 'Predicted:' in line]
    if hw_pred_str:
        hw_pred = int(hw_pred_str[0].split(',')[0].split(':')[1].strip())
        print(f"my_digit.png -> HW prediction: {hw_pred}")
    else:
        print("Failed to get prediction for my_digit.png")
        sys.exit(1)

if __name__ == "__main__":
    main()
