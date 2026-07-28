`include "bnn_pkg.sv"

// Top-level BNN inference core.
// Pipeline (sequential layers):
//   image_in (784 bits) -> [layer1 hidden] -> [layer2 hidden] -> [layer3 output z] -> [argmax] -> digit
// `start` begins processing; `done` pulses for one cycle with the result on `digit`.
// Pure registered popcount + integer compare => bit-exact vs the SW popcount recompute.
//
// NOTE: the output-layer argmax is inlined here (not a submodule) because iverilog
// does not reliably pass packed-array ports (z_out) between modules — reading the
// local z3 array directly avoids the X-propagation bug.

module bnn_top (
    input  logic               clk,
    input  logic               rst,
    input  logic               start,
    input  logic [(bnn_pkg::NIN1-1):0]  image_in,
    output logic [3:0]         digit,
    output logic               done
);

    import bnn_pkg::*;

    logic [(NOUT1-1):0] a1;
    logic [(NOUT2-1):0] a2;
    logic signed [31:0] z3 [0:(NOUT3-1)];
    logic [(NOUT3*32-1):0] z_flat_w;

    logic l1_done, l2_done, l3_done;
    logic am_start, am_done;

    bnn_layer #(.Nin(NIN1), .Nout(NOUT1), .IS_OUTPUT(1'b0)) L1 (
        .clk(clk), .rst(rst), .start(start), .act_in(image_in),
        .act_out(a1), .z_flat(), .done(l1_done));

    bnn_layer #(.Nin(NIN2), .Nout(NOUT2), .IS_OUTPUT(1'b0)) L2 (
        .clk(clk), .rst(rst), .start(l1_done), .act_in(a1),
        .act_out(a2), .z_flat(), .done(l2_done));

    bnn_layer #(.Nin(NIN3), .Nout(NOUT3), .IS_OUTPUT(1'b1)) L3 (
        .clk(clk), .rst(rst), .start(l2_done), .act_in(a2),
        .act_out(), .z_flat(z_flat_w), .done(l3_done));

    // Unpack flattened z vector into the z3 array for the argmax.
    genvar gi;
    generate
        for (gi = 0; gi < NOUT3; gi = gi + 1) begin : unpk
            assign z3[gi] = $signed(z_flat_w[gi*32 +: 32]);
        end
    endgenerate

    assign am_start = l3_done;

    // ---- Inlined output-layer argmax (scale_c * z_c + offset_c, argmax) ----
    `ifndef MEM_PATH
        `define MEM_PATH "mem_files/"
    `endif
    reg [63:0] scale_raw [0:(NOUT3-1)];
    reg [63:0] offset_raw [0:(NOUT3-1)];
    initial begin
        $readmemh({`MEM_PATH, "layer3_scales.hex"},  scale_raw);
        $readmemh({`MEM_PATH, "layer3_offsets.hex"}, offset_raw);
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            digit <= 0;
            am_done <= 1'b0;
        end else begin
            am_done <= 1'b0;
            if (am_start) begin
                real best;
                int  best_idx;
                best = -1.0e30;
                best_idx = 0;
                for (int c = 0; c < NOUT3; c = c + 1) begin
                    real sc, of, lg;
                    sc = $bitstoreal(scale_raw[c]);
                    of = $bitstoreal(offset_raw[c]);
                    lg = sc * $itor(z3[c]) + of;
                    if (lg > best) begin
                        best = lg;
                        best_idx = c;
                    end
                end
                digit <= best_idx[3:0];
                am_done <= 1'b1;
            end
        end
    end

    assign done = am_done;

    `ifdef DEBUG
    always @(posedge done) begin
        $display("[top] in=%h", image_in);
        $display("[top] a1=%h", a1);
        $display("[top] a2=%h", a2);
        for (int c = 0; c < NOUT3; c = c + 1)
            $display("[top] z3[%0d]=%0d", c, $signed(z3[c]));
        $display("[top] digit=%0d", digit);
    end
    `endif

endmodule
