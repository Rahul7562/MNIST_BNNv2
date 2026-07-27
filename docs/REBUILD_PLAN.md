# MNIST_BNNv2 — Rebuild Execution Plan

Director: Nemo (orchestrates, reviews, merges — never writes implementation code).
Implementer: Google Jules (creates sessions on the `MNIST_BNNv2` GitHub source,
opens PRs; Director gated-reviews and merges).

## Hard rules
- Jules MCP tools are driven via the REST API (terminal) because the MCP toolset is
  not registered in this Hermes session. Same `jules_mcp_server.py` dispatch.
- Every Jules session: `source = sources/github/Rahul7562/MNIST_BNNv2`,
  `startingBranch = finalized`, `automationMode = AUTO_CREATE_PR`,
  `requirePlanApproval = true` (Director gates the plan before any code is written).
- Reference code in the repo is **reference only**; rebuild cleanly per
  `docs/ARCHITECTURE.md`. Never hardcode paths; config-driven everywhere.
- A PR is merged only after Director review against the code-review rubric and after
  its tests pass locally.

## Phase 1 — Software ML pipeline  (Jules session A)
Scope: dataset + preprocessing, model, training, binarization, export, `convert_image`.
Deliverables: `sw/`, `scripts/train.py`, `scripts/export.py`, `scripts/convert_image.py`,
`configs/*.yaml`, `tests/test_model.py`, `tests/test_training.py`, `tests/test_export.py`,
`tests/test_preprocess.py`.
Acceptance: >95% test acc (aim 98%+); `mem_files/` export reconstructs inference
bit-exact vs torch model on ≥100 samples; `convert_image` correct on known digit;
pytest green; lint clean.

## Phase 2 — FPGA hardware + simulation  (Jules session B, after A merges)
Scope: `hw/rtl/*`, `hw/sim/bnn_tb.sv`, `hw/sim/run_sim.sh`, `scripts/verify_hw.py`,
`tests/test_hw.py`, `configs/hardware.yaml`.
Acceptance: compiles in Icarus + Verilator; 100% match vs SW popcount recompute on
≥40 images incl `my_digit.png`; parameterized, FSM with done; no hardcoded paths.

## Phase 3 — Vivado integration  (Jules session C, after B)
Scope: Vivado project Tcl, XDC timing/pin constraints (Zedboard), synth/impl run
scripts, resource/timing report parsing.
Acceptance: project builds; timing closure at target clock; BRAM/LUT/DSP report within
budget; bitstream-capable flow (generation optional if no board license).

## Phase 4 — Docs, CI, final validation  (Jules session D)
Scope: `docs/` (how-to train/export/sim/synth), CI workflow (train→export→sim→synth),
one-command retrain+regenerate, final validation vs `my_digit.png` + held-out set,
performance analysis.
Acceptance: CI green end-to-end; `my_digit.png` classified correctly; reproducible
build from clean checkout.

## Parallelization note
Phase 1 is a single coherent SW session (dataset format and export contract are
tightly coupled — splitting would cause cross-session drift). HW/CI sessions depend
on the exported contract, so they run strictly after Phase 1 merges. Docs/CI can
partially parallelize with Phase 3 once the interface is frozen.
