`ifndef BNN_PKG_SV
`define BNN_PKG_SV

// Global parameters and typedefs for the MNIST BNN hardware inference core.
// All layer dimensions come from the export contract (mem_files/export_meta.json):
//   layer1: 784 -> 256   (hidden, XNOR-popcount + integer threshold)
//   layer2: 256 -> 256   (hidden, XNOR-popcount + integer threshold)
//   layer3: 256 -> 10    (output, z = 2*P - N, logit = scale*z + offset, argmax)
//
// Bit convention (ARCHITECTURE.md §5): bits are {0,1} with 1 <=> value +1.
//   XNOR popcount P = popcount(~(a ^ w)).  True dot z = 2*P - N.
//   Hidden decision: a' = +1 iff z > T  <=>  P >= Th,  Th = floor((T+N)/2)+1.

package bnn_pkg;
    localparam int NIN1  = 784;
    localparam int NOUT1 = 256;
    localparam int NIN2  = 256;
    localparam int NOUT2 = 256;
    localparam int NIN3  = 256;
    localparam int NOUT3 = 10;
    localparam int NUM_CLASSES = 10;

    // Max width for threshold / popcount accumulators.
    localparam int TH_W = 16;          // threshold magnitude (< N <= 784)
    localparam int Z_W  = 32;          // z = 2*P - N  (fits in signed 32)
endpackage

`endif
