import os
import math
from pathlib import Path

def process_thresholds():
    # Fix path depending on where the script is run
    if Path("../mem_files").exists():
        mem_dir = Path("../mem_files")
    elif Path("mem_files").exists():
        mem_dir = Path("mem_files")
    else:
        mem_dir = Path("/app/mem_files")

    # Layer 1
    with open(mem_dir / "layer1_thresholds.mem", "r") as f:
        l1_floats = [float(x.strip()) for x in f.readlines()]
    with open(mem_dir / "layer1_thresholds_int.mem", "w") as f:
        for val in l1_floats:
            int_thresh = math.ceil((val + 784.0) / 2.0)
            f.write(f"{int_thresh:x}\n")

    # Layer 2
    with open(mem_dir / "layer2_thresholds.mem", "r") as f:
        l2_floats = [float(x.strip()) for x in f.readlines()]
    with open(mem_dir / "layer2_thresholds_int.mem", "w") as f:
        for val in l2_floats:
            int_thresh = math.ceil((val + 256.0) / 2.0)
            f.write(f"{int_thresh:x}\n")

    # Layer 3
    with open(mem_dir / "layer3_thresholds.mem", "r") as f:
        l3_floats = [float(x.strip()) for x in f.readlines()]
    with open(mem_dir / "layer3_thresholds_int.mem", "w") as f:
        for val in l3_floats:
            int_val = int(round(val * 128.0))
            if int_val < 0:
                int_val = (1 << 32) + int_val
            f.write(f"{int_val:08x}\n")

if __name__ == "__main__":
    process_thresholds()
