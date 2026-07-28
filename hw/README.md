# Phase 2 — FPGA Hardware (SystemVerilog) + Icarus Simulation

This directory contains the FPGA inference core for the MNIST BNN and its Icarus
Verilog simulation. The RTL is **bit-exact** with the software popcount inference
described in `docs/ARCHITECTURE.md` §5.

## Layout

```
hw/
  rtl/
    bnn_pkg.sv      parameters + typedefs (layer sizes, NUM_CLASSES)
    popcount.sv     population-count function (width-parameterized, X-safe)
    bnn_layer.sv    one BNN layer (XNOR-popcount, registered, sequential)
    bnn_top.sv      top-level: L1 -> L2 -> L3 -> argmax -> digit
  sim/
    bnn_tb.sv       testbench: drives test vectors, checks HW == SW popcount
    run_sim.sh      build + run (iverilog + vvp)
  mem/              generated at sim time (see scripts/prepare_hw_mem.py)
    layer{1,2,3}_weights.mem   big-endian hex weight bits (bit1 = +1)
    layer{1,2}_thresholds.hex integer thresholds (hex)
    layer3_scales.hex         IEEE-754 double (per-class logit scale)
    layer3_offsets.hex        IEEE-754 double (per-class logit offset)
    testvecs.mem              input activation bits (784-bit hex per image)
    testlabels.mem            expected digit (SW popcount recompute); 255 = skip
```

## Bit-significance convention (IMPORTANT)

All 784/256-bit vectors are stored **MSB-first**:
- `image_in[783]` = pixel 0  (set by `$fscanf`/`$readmemh`, which load hex MSB-first)
- `wmem[j][N-1]` = weight row j, bit 0
- `act_out[Nout-1-j]` = activation of neuron j  (see `bnn_layer.sv`)

Because `$readmemh` and `$fscanf` both load hex strings MSB-first, and activations
are written MSB-first (`act_out[Nout-1-j]`), the XNOR-popcount pairing
`~(act_in ^ wmem[j])` is consistent across every layer. **Do not change this
convention** without also flipping the weight/input load order — mixing LSB-first
registers with MSB-first hex loads was the root cause of an earlier 47/51 failure.

## Math (matches ARCHITECTURE §5)

- `P_j = popcount(XNOR(act_in, w_j))`, `z_j = 2*P_j - N`
- hidden: `a'_j = (P_j >= Th_j)`, `Th_j = floor((T_j + N)/2) + 1`
- output (L3): `z_j` forwarded; `logit_c = scale_c * z_c + offset_c`; `digit = argmax_c`

## Build & run (simulation only — no Vivado needed)

Requires `iverilog` + `vvp` (`apt install iverilog`). The sim prepares `hw/mem/`
from the exported parameters in `mem_files/` plus MNIST test images and
`my_digit.png`.

```bash
bash hw/sim/run_sim.sh
```

This will:
1. Run `scripts/prepare_hw_mem.py` to generate `hw/mem/` (hex weights, hex
   thresholds, IEEE-754 scales/offsets, and test vectors from `Dataset/t10k-*`
   + `my_digit.png`).
2. Compile the RTL with `iverilog -g2012`.
3. Run the testbench with `vvp`.

Expected output (SIM_COUNT=50 default):

```
Loaded 51 test vectors
vec 50 (my_digit.png): pred=N (label unknown, cross-check with SW)
========================================
Checked (labeled) vectors: 50
Mismatches: 0
HW SIM PASS: 100% match on >=40 images
```

## Verification gate

The testbench checks **HW popcount inference == SW popcount recompute** for every
labeled vector (≥40 required, including `my_digit.png`). The expected digit is
computed by `scripts/prepare_hw_mem.py` using the exact exported `.mem` files, so
the gate is self-contained and does not depend on the dataset label (which can
differ from the model's own prediction).

Reproduce the full 10k SW/HW cross-check independently with:

```bash
python3 scripts/verify_phase1.py     # SW vs HW popcount over 10k (100% bit-exact)
python3 scripts/convert_image.py     # my_digit.png: SW=N, HW=N, OK
```

## Vivado (Phase 3 — not yet wired)

The RTL uses synthesizable constructs (`always_ff`, `$readmemh` for BRAM init,
registered popcount). Phase 3 will add the Xilinx project wrapper, timing
constraints, and BRAM inference directives. The simulation here is the functional
gate; synthesizability is validated separately in Phase 3.

## Known limitations (sim only)

- The output argmax uses Verilog `real` (`$bitstoreal`) for the scale/offset
  multiply. This is for functional verification only; a real FPGA build uses
  fixed-point/DSP (Phase 3). The arithmetic is bit-exact with the SW reference.
- Icarus cannot pass packed-array ports reliably between modules; `z_out` is
  flattened to `z_flat` (see `bnn_layer.sv` / `bnn_top.sv`). Keep this if you
  add layers.
