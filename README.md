# MNIST_BNNv2

This project is an FPGA implementation of a Binarized Neural Network (BNN) for recognizing handwritten digits from the MNIST dataset.

## Target Device and IO Standards

The default target device is a Xilinx Zynq-7000 SoC (xc7z020clg484-1), as found on the ZedBoard.

### IO Requirements
If you decide to port this design to another board, please be aware of the following IO standards:
- **`clk`**: 100 MHz clock input. Requires `LVCMOS33` (3.3V) standard in the default constraints.
- **`led[3:0]`**: 4-bit output representing the predicted digit. Requires `LVCMOS33` (3.3V) standard.
- **`done`**: 1-bit output signaling computation completion. Requires `LVCMOS33` (3.3V) standard.
- **`rst`**: Active-high reset. Requires `LVCMOS18` (1.8V) standard in the default constraints (mapped to BTNU on the ZedBoard).
- **`start`**: Active-high start signal. Requires `LVCMOS18` (1.8V) standard in the default constraints (mapped to BTNR on the ZedBoard).

**Note:** If porting to a board with only 3.3V IO banks (e.g., Basys 3), you must update the constraints (`.xdc` file) for `rst` and `start` to use `LVCMOS33` instead of `LVCMOS18` to avoid Design Rule Check (DRC) errors during implementation.

## Project Structure
- `project_1/project_1.srcs/sources_1/new/bnn_top.sv`: The top-level module containing the sequential state machine.
- `project_1/mem_files/`: Memory initialization files (.mem) containing the BNN weights, biases, and the input image.
- `watch_my_digit.ps1`: A PowerShell script to continuously convert an input image (`my_digit.png`) into a `.mem` file for simulation or synthesis.
- `project_1/recreate_project_legacy.tcl`: A TCL script to recreate the Vivado project.

## How to build
To create the Vivado project, you can run the provided TCL script:
```
vivado -mode batch -source project_1/recreate_project_legacy.tcl -tclargs <new_project_name> <part>
```
If not specified, the project name defaults to `project_1_legacy` and the part defaults to `xc7z020clg484-1`.
