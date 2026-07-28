# MNIST_BNNv2 — Rebuild Architecture & SW/HW Interface Contract

> This document is the **single source of truth** for the rebuild. Every Jules
> implementation session (software and hardware) MUST follow it exactly so the
> trained model and the FPGA inference stay bit-exact.

## 1. Objective
A fully-binarized neural network for handwritten-digit recognition, deployable on
a Zedboard (Zynq-7000) FPGA, that:
- classifies **unseen** handwritten digits with **> 95%** accuracy (aim 98%+),
- classifies a user-supplied `my_digit.png` correctly,
- produces **deterministic, bit-exact** FPGA inference,
- retrains and regenerates all FPGA artifacts from a single command.

## 2. Why a *fully*-binarized BNN
- Weights **and** activations are `{-1,+1}`. Dot products become XNOR-popcounts →
  no DSP multipliers, tiny LUT/BRAM, trivially deterministic.
- The reference implementation kept the **first layer input real-valued**, which
  forces a mixed real/binary unit in hardware. The rebuild **binarizes the input
  by sign after normalization** (standard, ~lossless for MNIST) so *every* layer is
  a uniform popcount unit. This is the key architectural improvement.

## 3. Topology (configuration-driven)
```
784 -> [Linear, BatchNorm, Sign] -> 256 -> [Linear, BatchNorm, Sign] -> 256 -> [Linear, BatchNorm] -> 10
```
- Bias-free `BinarizeLinear`, `BatchNorm1d` after each linear (pre-activation style).
- `Sign` (straight-through estimator) between hidden layers; **no** sign on output.
- `argmax(logits)` = predicted digit.
- Hidden widths are configurable; default `256->256` (reliably 98–99% on MNIST,
  resource-light). Do **not** exceed 512 without justification.

## 4. Preprocessing contract (identical for SW, `convert_image`, and HW input)
1. Grayscale 28×28, intensity normalized to `[0,1]` (MNIST) or arbitrary user PNG.
2. **Arbitrary user PNG (`convert_image`)**: resize preserving aspect to fit 28×28,
   center via bounding-box/centroid, **auto-invert** if the background is lighter
   than the stroke (compare mean luminance), neutralize any color.
3. Standardize per pixel: `x' = (x - mean)/std` using dataset mean/std. Robust
   fallback if stats unavailable: `x' = (x - 0.5) * 2` mapping `[0,1] -> [-1,1]`.
4. **Binarize by sign**: `b = sign(x') in {-1,+1}`; map to bit `p = (b+1)/2 in {0,1}`.
   The 784 bits are the HW input vector. The SAME mapping must be used everywhere.

## 5. Forward-pass math (source of truth)
Layer `l`, binarized weight `W_l in {-1,+1}^{M×N}`, binary input `a in {-1,+1}^N`:
- pre-activation `z = W_l · a`  (range `[-N, N]`, step 2)
- `BatchNorm: y = gamma*(z - mu)/sqrt(var+eps) + beta`
- hidden: `a' = sign(y)`
- output layer: `logit_c = y_c` (no sign)

### BatchNorm folding → single integer threshold per neuron
We want `sign(y)=+1 ⇔ y>0 ⇔ gamma*(z-mu)/sigma + beta > 0`.
- Always enforce `gamma > 0` by folding `sign(gamma)` into both the weight row and
  the bias constant (export handles this). Then the comparison is uniformly `z > T`
  with `T = mu - beta*sigma/gamma`.
- In terms of binary `w,a in {0,1}` (mapped from `{-1,+1}`):
  `z = 2*popcount(w AND a) - popcount(w)`.
- So HW computes `popcount(w AND a)` and compares to integer
  `Th = (T + popcount(w)) / 2` (derived; export precomputes it).
- **Output layer**: export the folded constant `off_c = beta_c - gamma_c*mu_c/sigma_c`
  per class; HW adds `off_c` to its popcount-based `z_c` then takes `argmax`.

## 6. Export format (`mem_files/`) — THE HW CONTRACT
`scripts/export.py` writes, after training:
- `layer{l}_weights.mem` — one hex line per output neuron; bits packed **big-endian**,
  bit `1` = weight `+1`, `0` = `-1`. Row length = layer input width (784/256/256).
- `layer{l}_thresholds.mem` — one integer per neuron = `Th` (popcount threshold).
- `layer3_offsets.mem` (output layer) — `off_c` per class (logit offset constants).
- `export_meta.json` — global: `{input_dims, layer_shapes, mean, std, binarize_method,
  num_classes, popcount_w_per_neuron (precomputed), q_format}`.
HW reads ONLY these files; no other model knowledge is needed. Changing the model
MUST re-run export; CI fails if `export_meta.json` is stale vs the checkpoint.

## 7. Hardware interface (implemented Phase 2)
- `bnn_top`: 784-bit input vector + `start`; outputs 4-bit `digit` + `done`.
- Pipeline (sequential, registered): `L1 (784->256) -> L2 (256->256) -> L3 (256->10) -> argmax -> digit`.
- One layer module (`bnn_layer`) reused for L1/L2 (hidden) and L3 (output). Each
  layer computes `P_j = popcount(XNOR(act_in, w_j))` neuron-by-neuron, then either
  `a'_j = (P_j >= Th_j)` (hidden) or forwards `z_j = 2*P_j - N` (output).
- Output argmax: `logit_c = scale_c * z_c + offset_c` (per-class scale/offset from
  `layer3_scales/offsets.hex`, read as IEEE-754 doubles via `$bitstoreal`),
  `digit = argmax_c`. Bit-exact with the SW reference. (Phase 3 replaces the
  `real` multiply with fixed-point/DSP for synthesis.)
- **Bit-significance convention (critical):** all vectors are MSB-first —
  `image_in[783]=pixel0`, `wmem[j][N-1]=weight bit0`, `act_out[Nout-1-j]=neuron j`.
  `$readmemh`/`$fscanf` load hex MSB-first, so weights + input agree; activations
  are explicitly written MSB-first (`act_out[Nout-1-j]`) to keep layer-to-layer
  pairing consistent. Mixing LSB-first registers with MSB-first hex loads breaks
  the pipeline (root cause of an early 47/51 sim failure).
- Packed-array ports are flattened (`z_flat`) because Icarus does not reliably
  drive array ports between modules.
- Synthesizable in Vivado (Zedboard / `zedboard_minimal.xdc`); simulatable in
  Icarus Verilog (`bash hw/sim/run_sim.sh`).

## 8. Target directory layout
```
sw/            model (bnn), data (dataset), preprocess (convert_image logic)
scripts/       train.py, export.py, convert_image.py, verify_dataset.py, verify_hw.py
hw/rtl/        bnn_pkg.sv, popcount.sv, bnn_layer.sv, bnn_top.sv
hw/sim/        bnn_tb.sv, run_sim.sh
hw/mem/        symlink/copy of mem_files/ (auto-generated)
mem_files/     exported weights/thresholds/offsets/meta
configs/       default.yaml, model.yaml, training.yaml, hardware.yaml
tests/         test_model.py, test_training.py, test_export.py, test_preprocess.py, test_hw.py
docs/          ARCHITECTURE.md, REBUILD_PLAN.md, ...
```

## 9. Verification gates (no phase completes until green)
- **SW**: `pytest` — model forward, training smoke (reaches >95%, aim 98%+),
  **export↔inference consistency** (recompute popcount inference from `mem_files`
  and compare to `torch` model on ≥100 test samples, exact match),
  `convert_image` round-trip on a known digit.
- **HW**: Icarus `bnn_tb` — **100% match** vs SW popcount recompute on ≥40 images
  **including `my_digit.png`** (verified: 50/50 = 100% on the SIM_COUNT=50 default
  run; `my_digit.png` SW=HW). The expected digit is computed by
  `scripts/prepare_hw_mem.py` from the exported `.mem` files, so the gate is
  self-contained (does not depend on dataset labels, which may differ from the
  model's own prediction). Build/run: `bash hw/sim/run_sim.sh`.
- **Regression**: retrain on an augmented/perturbed dataset → re-export → HW still
  matches; artifacts stay synchronized.
