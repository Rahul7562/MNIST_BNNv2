# MNIST_BNNv2

A fully-binarized fully-connected BNN (784→256→256→10) trained from scratch and
deployed as an FPGA inference core. Goal: **>95% accuracy on unseen handwritten
digits**, with **bit-exact** agreement between software inference and the hardware
popcount path.

## Repository layout

| Path | Contents |
|------|----------|
| `sw/model/bnn.py` | BNN (per-pixel standardized input → Sign binarize; BinarySign hidden acts) |
| `sw/training/dataset.py` | MNIST loading + per-pixel standardization |
| `scripts/train.py` | Reproducible training (seeded) → saves checkpoint + mean/std |
| `scripts/export.py` | XNOR-popcount parameter export (ARCHITECTURE.md §5) |
| `scripts/convert_image.py` | Classify `my_digit.png` (SW + HW popcount) |
| `scripts/verify_phase1.py` | Independent 10k SW/HW bit-exact gate |
| `mem_files/` | Exported weights / thresholds / scales / offsets / meta (source of truth) |
| `configs/` | `default.yaml`, `model.yaml`, `training.yaml` |
| `hw/rtl/` | SystemVerilog core: `bnn_pkg`, `popcount`, `bnn_layer`, `bnn_top` |
| `hw/sim/` | Icarus testbench + `run_sim.sh` (HW == SW gate) |
| `hw/constraints/zedboard.xdc` | Zedboard (xc7z020clg400-1) constraints |
| `hw/vivado/` | `build.tcl`, `run_vivado.sh`, `README.md` (Vivado build guide) |
| `docs/` | `ARCHITECTURE.md` (math + HW contract), `PROJECT_ROADMAP.md` |
| `tests/` | `test_model.py`, `test_training.py`, `test_export.py` |

## Quick start (software + simulation)

```bash
pip install -r requirements.txt

python scripts/train.py                 # train + export mem_files/   (reproducible)
bash hw/sim/run_sim.sh                  # Icarus sim: HW == SW popcount (100%)
python scripts/verify_phase1.py 10000    # independent 10k gate
pytest tests/                           # unit tests (7/7)
python scripts/convert_image.py         # classify my_digit.png (SW + HW)
```

## Building on FPGA with Vivado

Full walkthrough (GUI + batch) is in **`hw/vivado/README.md`**. Summary:

```bash
bash hw/vivado/run_vivado.sh           # synth + impl + bitstream (Vivado on PATH)
```

This produces `bnn_top.bit` plus utilization/timing reports for the Zedboard. The
same RTL that passes the Icarus simulation is what gets synthesized — no separate
"sim vs synth" core. The core loads weights/thresholds/scales from `hw/mem/` (generated
by the sim script) via `$readmemh`; no manual `.coe` is needed.

## Verification status

- SW accuracy: **96.45%** • HW(popcount): **96.45%** • bit-exact HW↔SW: **100%** (10k)
- Icarus sim: **100%** (≥40 MNIST images incl. `my_digit.png`)
- CI (`.github/workflows/ci.yml`): pytest + sim + 10k gate on every push/PR

## Reproducibility

Training is seeded (`configs/training.yaml`), and the export pipeline regenerates all
FPGA parameters from the checkpoint, so **retraining on a new dataset and regenerating
FPGA artifacts requires only**: `python scripts/train.py` → `bash hw/sim/run_sim.sh`
→ (on a Vivado host) `bash hw/vivado/run_vivado.sh`.

See `docs/ARCHITECTURE.md` for the math contract and `hw/README.md` for the HW
build/sim flow.
