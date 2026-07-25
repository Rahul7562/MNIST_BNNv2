# FPGA Architecture Design

## Overview
The hardware inference engine implements a streaming Convolutional Neural Network (CNN) specifically tailored for Binarized Neural Networks (BNNs).

## Core Building Blocks
### XNOR-Popcount MAC
Standard Convolutional networks require floating-point or integer Multiplier-Accumulator (MAC) operations. By constraining weights and activations to binary values (-1 or 1), the multiplication becomes an XNOR gate, and the accumulation becomes a population count (popcount).

This allows the FPGA to execute thousands of MAC operations in parallel using basic LUT resources.

### Quantization & Batch Norm Folding
To avoid calculating floating-point Batch Normalization during inference, the `export.py` script analytically calculates the exact integer popcount threshold that causes the output to flip from -1 to 1. The FPGA simply compares the popcount result against this integer threshold (`quant_layer.sv`).

### Streaming Architecture
Rather than spatially unrolling the entire network (which would exceed Zynq-7000 resource limits), the design utilizes a **Streaming Line-Buffer Architecture**.
1. **Line Buffers (`line_buffer_3x3.sv`)**: Input pixels stream in 1-by-1. The line buffer retains the last few rows to output a 3x3 sliding window every clock cycle.
2. **Parallel Filters (`conv2d_bnn.sv`)**: While pixels process sequentially, all output channels (filters) for that pixel calculate in parallel.

## Throughput and Latency
The network operates at an expected 100MHz clock. Processing one 28x28 image requires streaming 784 pixels.
- **Latency**: ~800 clock cycles per image inference.
- **Throughput**: Since the architecture is pipelined, it can process one image per inference run (at a rate of 1 pixel/cycle stream).
