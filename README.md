# MNIST_BNNv2

A fully-binarized fully-connected BNN (784→256→256→10) trained from scratch and
deployed as an FPGA inference core. Goal: **>95% accuracy on unseen handwritten
digits**, with **bit-exact** agreement between software inference and the hardware
popcount path.

## What's here

| Path | What |
|------|------|
| `sw/model/bnn.py` | BNN (per-pixel standardized input → Sign binarize; BinarySign hidden acts) |
| `scripts/train.py` | Reproducible training (seeded), exports mem_files |
| `scripts/export.py` | XNOR-popcount parameter export (ARCHITECTURE.md §5) |
| `scripts/convert_image.py` | Classify `my_digit.png` (SW + HW popcount) |
| `scripts/verify_phase1.py` | Independent 10k SW/HW bit-exact gate |
| `hw/rtl/` | SystemVerilog inference core (`bnn_top`, `bnn_layer`, `popcount`) |
| `hw/sim/` | Icarus testbench + `run_sim.sh` (HW==SW gate) |
| `hw/constraints/zedboard.xdc` | Zedboard (xc7z020clg400-1) constraints |
| `hw/vivado/` | `build.tcl` + `run_vivado.sh` (synth/impl/bitstream) |
| `docs/ARCHITECTURE.md` | Frozen math contract + HW interface |
| `PROJECT_ROADMAP.md` | Phase status |

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py                 # train + export mem_files/
bash hw/sim/run_sim.sh                  # Icarus sim: HW == SW popcount (100%)
python scripts/verify_phase1.py 10000    # independent 10k gate (SW 96.45%, HW 96.45%, 100% exact)
pytest tests/                           # unit tests (7/7)
python scripts/convert_image.py         # classify my_digit.png
bash hw/vivado/run_vivado.sh           # (Vivado host) synthesize + bitstream
```

## Verification status

- SW accuracy: **96.45%** • HW(popcount): **96.45%** • bit-exact HW↔SW: **100%** (10k).
- Icarus sim: **100%** (≥40 MNIST images incl. `my_digit.png`).
- CI: `.github/workflows/ci.yml` runs pytest + sim + 10k gate on every push/PR.
- FPGA synthesis/bitstream: run `hw/vivado/run_vivado.sh` on a Vivado-equipped host
  (Vivado is not available in the sim-only CI environment).

See `docs/ARCHITECTURE.md` for the math and `hw/README.md` for the HW build/sim flow.
