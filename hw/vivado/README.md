# Using this project on Xilinx Vivado

This guide builds the MNIST BNN inference core (`bnn_top`) into a bitstream for the
Zedboard (xc7z020clg400-1). The exact same RTL that passes the Icarus simulation
is the RTL that is synthesized — there is no separate "sim vs synth" core.

> **Prerequisites**
> - Vivado (2022.1+ recommended; WebPACK edition is sufficient for the Zedboard).
>   Download: https://www.xilinx.com/products/design-tools/vivado.html
> - The exported parameters in `hw/mem/` (run `bash hw/sim/run_sim.sh` once, or
>   `python3 scripts/prepare_hw_mem.py`, to generate them from `mem_files/`).
> - A Zedboard (or any xc7z020clg400-1 board) and a micro-USB cable for JTAG.

---

## Option A — Batch build (one command)

```bash
# from the repo root
bash hw/vivado/run_vivado.sh
```

`run_vivado.sh` ensures `hw/mem/` exists, then invokes:

```tcl
vivado -mode batch -source hw/vivado/build.tcl -tclargs hw/mem
```

`build.tcl` performs, in order: `create_project` → `synth_design` →
`opt_design`/`place_design`/`route_design` → `write_bitstream`, and writes:

- `vivado_project/mnist_bnn_vivado.runs/impl_1/bnn_top.bit` — the bitstream
- `vivado_project/utilization.rpt` — LUT / FF / BRAM / DSP usage
- `vivado_project/timing_summary.rpt` — WNS / WNS slack, clock summary

On success the script prints the bitstream path and report locations.

---

## Option B — Vivado GUI (step by step)

1. **Launch Vivado** and choose *Create Project*.
   - Project name: `mnist_bnn_vivado`, location: anywhere.
   - Type: *RTL Project*, tick *Do not specify sources at this time*.
   - Default Part: search `xc7z020clg400-1` (Zedboard), select it, Finish.

2. **Add sources** — *Add or create design sources*:
   - Add all four RTL files from `hw/rtl/`:
     `bnn_pkg.sv`, `popcount.sv`, `bnn_layer.sv`, `bnn_top.sv`.
   - Set `bnn_top` as the **top module** (Vivado detects it automatically).
   - In *Project Settings → Synthesis*, add `hw/rtl` to **Include Paths**, and add
     theVerilog Define: `MEM_PATH=hw/mem/` (including the trailing slash). This tells
     the `$readmemh` calls where the weight/threshold/scale `.hex`/`.mem` files live.

3. **Add constraints** — *Add or create constraints*:
   - Add `hw/constraints/zedboard.xdc`. This pins the 100 MHz clock (Y9), reset,
     `start`, `done`, and the 4-bit `digit` output (to LEDs LD0–LD3).

4. **Run Synthesis** — *Flow → Synthesis*. Wait for *Synthesis Complete*.

5. **Run Implementation** — *Flow → Implementation*. Check the
   *Timing Summary* (WNS should be positive at 100 MHz) and *Utilization* reports.

6. **Generate Bitstream** — *Flow → Generate Bitstream*. Output:
   `mnist_bnn_vivado.runs/impl_1/bnn_top.bit`.

7. **(Optional) Open Hardware Manager** → *Auto Connect* → *Program Device* to
   flash the Zedboard over JTAG.

---

## How inference works on the board

`bnn_top` is a **registered, sequential pipeline**:

```
image_in[783:0] ──start──▶ L1 (784→256) ─▶ L2 (256→256) ─▶ L3 (256→10, z) ─▶ argmax ─▶ digit[3:0] ──done──▶
```

- Drive `image_in` with your 784-bit binarized image (bit = 1 ⇔ pixel ≥ 0 after
  standardization; MSB-first: `image_in[783]` = pixel 0).
- Pulse `start` for one clock.
- When `done` goes high (one cycle), `digit` holds the predicted class (0–9).
- All weights/thresholds/scales are loaded from `hw/mem/` at elaboration via
  `$readmemh` (Vivado initializes BRAM from these — no manual `.coe` needed).

### Classifying your own digit (`my_digit.png`)

The Python side does the standardization + binarization and (for the PC flow)
prints the expected digit:

```bash
python scripts/convert_image.py     # prints SW prediction and HW popcount prediction
```

For the FPGA, `scripts/prepare_hw_mem.py` already emits `hw/mem/testvecs.mem`
(including `my_digit.png` if present). To feed a single image to the board you would
serialize that 784-bit row onto `image_in` (e.g. from a BRAM/AXI-DMA in a larger
design); the standalone core here exposes `image_in` as a top-level port for
verification and bring-up.

---

## Notes & troubleshooting

- **`MEM_PATH` define is mandatory.** If you see `$readmemh: file not found`, the
  define points at the wrong directory. It must end with `/` (e.g. `hw/mem/`).
- **Resource usage** is dominated by the popcount/XNOR over 784×256 + 256×256
  weight bits (LUT/FF heavy; weights fit comfortably in BRAM). Exact numbers are in
  `utilization.rpt` after a build. No DSP blocks are required (the output argmax uses
  fixed-point integer arithmetic, `FXD=16`).
- **Timing**: the design is a slow sequential pipeline; 100 MHz closes easily on the
  Zedboard. If you push the clock higher, the critical path is the layer popcount
  adder tree — constrain `create_clock -period` in `zedboard.xdc` accordingly.
- **Changing the model**: re-run `python scripts/train.py` → `scripts/export.py` →
  `bash hw/sim/run_sim.sh` to regenerate `hw/mem/`, then re-synthesize. The hardware
  parameters are always synchronized with the trained model by construction.

See also: `hw/README.md` (sim flow), `docs/ARCHITECTURE.md` (math contract),
`PROJECT_ROADMAP.md` (status).
