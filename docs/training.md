# BNN Training Pipeline

## Architecture
The network is based on the Larq BNN tutorial, translating standard layers into their Quantized counterparts:
1. **Conv1**: 32 channels, 3x3 Kernel, MaxPool (2x2), BatchNorm
2. **Conv2**: 64 channels, 3x3 Kernel, MaxPool (2x2), BatchNorm
3. **Conv3**: 64 channels, 3x3 Kernel, BatchNorm
4. **FC1**: 64 Neurons, BatchNorm
5. **FC2**: 10 Neurons (Output), BatchNorm

## Straight-Through Estimator (STE)
During training, the `Binarize` autograd function applies a deterministic sign function to weights and activations. Because the derivative of `sign(x)` is zero almost everywhere, we employ a Straight-Through Estimator (STE) that simply passes the gradient through the quantization node unharmed, clipping at bounds `[-1, 1]`.

## Export to Hex (.mem)
To interface cleanly with SystemVerilog `$readmemh`, the exported values undergo:
1. Binary Mapping: `-1 -> 0` and `1 -> 1`
2. String Concatenation: Values flatten per channel.
3. Hex Conversion: 4 bits map to a single hex character.
