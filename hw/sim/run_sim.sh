#!/usr/bin/env bash
# Build + run the Icarus Verilog simulation of the BNN inference core.
# Requires: iverilog + vvp (apt install iverilog). Vivado not needed for sim.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

MEM_DIR="hw/mem"
SIM_DIR="hw/sim"

# 1) Generate hw/mem/ (hex weights, hex thresholds, IEEE-754 scales/offsets,
#    and test vectors + labels from MNIST + my_digit.png).
python3 scripts/prepare_hw_mem.py

# 2) Compile. MEM_PATH points the RTL $readmemh at hw/mem/.
mkdir -p build
iverilog -g2012 -Wall \
  -Ihw/rtl \
  -DMEM_PATH=\"${MEM_DIR}/\" \
  -o build/bnn_sim \
  hw/rtl/bnn_pkg.sv \
  hw/rtl/popcount.sv \
  hw/rtl/bnn_layer.sv \
  hw/rtl/bnn_top.sv \
  hw/sim/bnn_tb.sv

# 3) Run.
vvp build/bnn_sim
