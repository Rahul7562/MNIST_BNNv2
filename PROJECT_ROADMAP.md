# MNIST_BNNv2 Complete Rebuild — Project Roadmap

Fully-binarized FC-BNN (784→256→256→10) that runs on an FPGA and exceeds 95%
accuracy on unseen handwritten digits. Built from first principles against
`docs/ARCHITECTURE.md` (frozen math contract, §5). 

## Status (updated 2026-07-28)

| Phase | Scope | Status | Merge |
|-------|-------|--------|-------|
| **1 — SW ML pipeline** | Train (input binarization + Sign STE), export (XNOR popcount folding), `convert_image.py`, `verify_phase1.py`, tests | ✅ Merged | PR #8 → `finalized` |
| **2 — FPGA HW + Icarus sim** | `bnn_pkg/popcount/bnn_layer/bnn_top` (SystemVerilog), testbench, `prepare_hw_mem.py` | ✅ Merged | PR #9 → `finalized` |
| **3 — Vivado integration** | Synthesizable core (fixed-point argmax), Zedboard XDC, `build.tcl`/`run_vivado.sh` | ✅ Merged | PR #10 → `finalized` |
| **4 — Docs + CI + final validation** | CI workflow, consolidated docs, top-level README, final gate run | 🟡 In progress | — |

## Phase 1 — Software ML pipeline (MERGED #8)
- `sw/model/bnn.py`: per-pixel standardized input → Sign(STE) binarize; BinarySign
  hidden activations; latent weights clipped to [-1,1].
- `scripts/train.py`: seeded (reproducible), saves **full per-pixel** mean/std (the
  scalar-mean bug from the reference impl is fixed).
- `scripts/export.py`: exports weights/thresholds + output-layer scale/offset in the
  XNOR-popcount convention (`P=popcount(XNOR(w,a))`, `z=2P−N`, `logit=scale·z+off`).
- **Verification**: SW 96.45%, HW(popcount) 96.45%, 100% bit-exact over 10k samples.
  pytest 7/7.

## Phase 2 — FPGA HW + Icarus sim (MERGED #9)
- Generic sequential `bnn_layer` (XNOR popcount, thresholds), `bnn_top` pipeline.
- Bit-exact with SW popcount recompute: **100% (50/50)** on Icarus with ≥40 MNIST
  images + `my_digit.png`.
- **Key lesson (cost 47/51 sim failures)**: all vectors MSB-first
  (`image_in[783]=pixel0`, `wmem[j][N−1]=weight bit0`, `act_out[Nout−1−j]=neuron j`);
  activations must be written MSB-first so layer-to-layer pairing stays consistent.

## Phase 3 — Vivado integration (MERGED #10)
- `bnn_top` output argmax rewritten from `real`/`$bitstoreal` to **fixed-point
  integer** (FXD=16): `logit_fxd = scale_fxd·z + offset_fxd`. Removed the only
  non-synthesizable construct — the same core that passes Icarus is the one that
  synthesizes.
- `hw/constraints/zedboard.xdc` (xc7z020clg400-1, 100 MHz), `hw/vivado/build.tcl`,
  `hw/vivado/run_vivado.sh`.
- **Synthesis/PAR/bitstream require a Vivado-equipped host** (not available in the
  sim-only CI env). Run `bash hw/vivado/run_vivado.sh` there.

## Phase 4 — Docs + CI + final validation (IN PROGRESS)
- `.github/workflows/ci.yml`: installs iverilog + deps, runs pytest, Icarus sim,
  and the independent 10k SW/HW verification on every push/PR to `finalized`.
- This roadmap, top-level `README.md`, ARCHITECTURE.md / hw/README.md kept current.
- Final gate: re-run all verification; confirm `my_digit.png` classifies correctly
  on both SW and HW paths.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/train.py                 # re-train (reproducible seeds)
python scripts/export.py                # regenerate mem_files/
bash hw/sim/run_sim.sh                  # Icarus sim (HW == SW popcount)
python scripts/verify_phase1.py 10000   # independent 10k gate
pytest tests/                           # unit tests
python scripts/convert_image.py         # classify my_digit.png (SW + HW)
bash hw/vivado/run_vivado.sh           # (Vivado host) build bitstream
```
