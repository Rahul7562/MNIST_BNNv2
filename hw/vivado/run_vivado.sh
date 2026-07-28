#!/usr/bin/env bash
# Build the MNIST BNN core in Vivado (batch mode).
# Requires Vivado on PATH (NOT available in the CI/sim environment; this script is
# the handoff for the user's machine / a build server with Vivado installed).
#
# What it does:
#   1. Generates hw/mem/ from the exported parameters (iverilog not needed, but
#      prepare_hw_mem.py is pure Python + numpy/torch/Pillow).
#   2. Launches Vivado in batch mode with hw/vivado/build.tcl, which synthesizes,
#      implements, and writes the bitstream for the Zedboard (xc7z020clg400-1).
#
# Usage:
#   bash hw/vivado/run_vivado.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

MEM_DIR="hw/mem"
if [ ! -f "$MEM_DIR/layer1_weights.mem" ]; then
    echo "[run_vivado] generating hw/mem/ ..."
    python3 scripts/prepare_hw_mem.py
fi

PART="xc7z020clg400-1"
if ! command -v vivado >/dev/null 2>&1; then
    echo "ERROR: vivado not found on PATH. Install Vivado (WebPACK edition is free)"
    echo "        and ensure 'vivado' is on PATH, then re-run this script."
    exit 1
fi

echo "[run_vivado] launching Vivado batch build (part=$PART) ..."
vivado -mode batch -source hw/vivado/build.tcl -tclargs "$MEM_DIR"
echo "[run_vivado] done. Bitstream + reports under vivado_project/"
