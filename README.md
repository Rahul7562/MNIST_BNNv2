# MNIST Binarized Neural Network (BNN)

This repository contains a complete, production-ready Binarized Neural Network (BNN) for MNIST digit classification, targeted for deployment on Xilinx FPGAs. It achieves >97% test accuracy and implements a highly efficient, streaming Verilog inference engine.

## Directory Structure
- `src/`: Python scripts for training (`train.py`), export (`export.py`), verification (`verify.py`), and interactive UI (`interactive.py`).
- `hdl/`: SystemVerilog files for the streaming FPGA inference engine.
- `vivado/`: TCL build script and XDC constraints for Vivado projects.
- `docs/`: Comprehensive project documentation.
- `tests/`: Verification scripts and SystemVerilog testbenches.

## Pipeline Overview

1. **Python Training**: Uses PyTorch with a Straight-Through Estimator (STE) to train a 3-Conv, 2-Dense CNN with binarized weights (-1/1) and activations.
2. **Export**: Batch Normalization layers are mathematically folded into integer thresholds. Weights and thresholds are exported into Vivado-compatible hex `.mem` files.
3. **FPGA Inference**: A streaming, pipelined hardware architecture in SystemVerilog. Uses an efficient XNOR-popcount MAC, processing the 28x28 image continuously to generate predictions.
4. **Verification & UI**: Includes cosimulation tools to verify RTL against PyTorch, and a drawing application to evaluate real-time performance.

## Usage

### 1. Training and Export
Run the training script to train the model, save the weights, and automatically export `.mem` files.
```bash
python src/train.py
```
*(Requires PyTorch. The exported `.mem` files will be saved in `mem_files/`)*

### 2. Verify Output Data
Generate testbench data from the MNIST test set and ensure the PyTorch model behaves as expected:
```bash
PYTHONPATH=src python src/verify.py
```

### 3. Interactive Digit Recognition
Draw digits in an OpenCV window and get real-time classifications using the trained BNN weights:
```bash
PYTHONPATH=src python src/interactive.py
```

### 4. Create Vivado Project
Generate the Vivado project for Synthesis and Implementation on your target board (Default: Zedboard xc7z020clg484-1).
```bash
cd vivado
vivado -mode batch -source build.tcl
```

### 5. Running Simulations (SystemVerilog)
A testbench is provided in `tests/bnn_tb.sv` which loads `mem_files/` and verifies hardware correctness. Use your preferred simulator (Vivado Simulator, Icarus Verilog) to compile `hdl/*.sv` and `tests/bnn_tb.sv`.

## Hardware Architecture
The BNN translates multiplication and addition into bitwise XNOR and popcount operations. We implement a pipelined streaming layout where pixels from the 28x28 image are processed continuously through line-buffers, eliminating the need to store massive intermediate feature maps in RAM.

Please refer to the `docs/` directory for an in-depth breakdown of the architecture, design choices, and build instructions.
