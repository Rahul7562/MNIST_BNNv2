`ifndef POPCOUNT_SV
`define POPCOUNT_SV

// Population count (number of set bits) of a vector, over the low `width` bits.
// Combinational; iverilog-unrolled. Suitable for sim; for FPGA the synthesis
// tool will map this to LUT-based popcount primitives.
//
// NOTE: `v` is declared wide (4096) so any layer width fits, but only the low
// `width` bits are valid; the caller must pass the true width so we do not count
// X-filled upper bits (iverilog would otherwise treat X as 1 and corrupt the count).
function automatic int popcount(input logic [(4096-1):0] v, input int width);
    int c;
    int i;
    c = 0;
    for (i = 0; i < width; i = i + 1) begin
        if (v[i]) c = c + 1;
    end
    return c;
endfunction

`endif
