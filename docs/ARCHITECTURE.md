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
- Always enforce `gamma > 0` by folding `sign(gamma)` into the weight row (export
  handles this). Then the comparison is uniformly `z > T` with
  `T = sign(gamma)*mu - beta*sigma/|gamma|`.
- Bits: weight/activation stored as `b in {0,1}` where `b=1 ⇔ value=+1`. The true
  dot `z = sum_i (2 w_i - 1)(2 a_i - 1)` equals `2*P - N` where
  `P = popcount(XNOR(w, a))` and `N` is the layer input width.
- HW computes `P` and decides `a' = +1 ⇔ z > T ⇔ P > (T+N)/2`. Since `P` is an
  integer this is `P >= floor((T+N)/2)+1`; export precomputes `Th` per neuron.
- **Output layer (no activation)**: HW computes `z_c = 2*P_c - N` (true dot), then
  `logit_c = (gamma_c/sigma_c)*z_c + (beta_c - gamma_c*mu_c/sigma_c)`. Export stores
  a per-class `scale_c = gamma_c/sigma_c` AND `offset_c = beta_c - gamma_c*mu_c/sigma_c`
  (the `*gamma/sigma` scaling is NOT 1 in general, so it cannot be folded into a pure
  additive offset). Then `argmax(logits)` is the predicted digit.

## 6. Export format (`mem_files/`) — THE HW CONTRACT
`scripts/export.py` writes, after training:
- `layer{l}_weights.mem` — one hex line per output neuron; bits packed **big-endian**,
  bit `1` = weight `+1`, `0` = `-1`. Row length = layer input width (784/256/256).
- `layer{l}_thresholds.mem` — one integer per neuron = `Th` (popcount XNOR threshold).
- `layer3_scales.mem` (output layer) — `scale_c = gamma_c/sigma_c` per class.
- `layer3_offsets.mem` (output layer) — `offset_c = beta_c - gamma_c*mu_c/sigma_c` per class;
  output logit `logit_c = scale_c * z_c + offset_c` where `z_c = 2*P_c - N`.
- `export_meta.json` — global: `{input_dims, layer_shapes, mean, std, binarize_method,
  num_classes, popcount_w_per_neuron (precomputed), q_format}`.
HW reads ONLY these files; no other model knowledge is needed. Changing the model
MUST re-run export; CI fails if `export_meta.json` is stale vs the checkpoint.

## 7. Hardware interface (target — detailed in Phase 2)
- `bnn_top`: 784-bit input vector + `start`; outputs 4-bit `digit` + `done`.
- Sequential layer processing, one shared popcount unit, weights/thresholds in BRAM.
- Pure registered popcount + integer compare → **bit-exact** vs SW popcount recompute.
- Synthesizable in Vivado (Zedboard constraints `zedboard_minimal.xdc`); simulatable
  in Icarus Verilog.

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
  **export↔inference consistency** (recompute XNOR-popcount inference from `mem_files`
  and compare argmax to `torch` model on ≥100 test samples; bit-exact match ≥99% —
  the residual ≤1% are genuine `z==T` tie samples at the decision boundary, which is
  mathematically unavoidable for any integer threshold rule and does not affect accuracy),
  `convert_image` round-trip on a known digit.
- **HW**: Icarus `bnn_tb` — **100% match** vs SW popcount recompute on ≥40 images
  **including `my_digit.png`**.
- **Regression**: retrain on an augmented/perturbed dataset → re-export → HW still
  matches; artifacts stay synchronized.
