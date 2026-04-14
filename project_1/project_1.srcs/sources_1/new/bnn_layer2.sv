module bnn_layer2 (
    input  logic [511:0] l1_in,
    output logic [255:0] l2_out
);

    reg [511:0] weights_l2 [0:255];
    reg [9:0]   thresh_l2  [0:255];
    reg [0:0]   invert_l2  [0:255];

    initial begin
        $readmemb("mem_files/weights_l2.mem", weights_l2);
        $readmemb("mem_files/thresh_l2.mem",  thresh_l2);
        $readmemb("mem_files/invert_l2.mem",  invert_l2);
    end

    genvar j;
    generate
        for (j = 0; j < 256; j = j + 1) begin : g_l2_neuron
            wire [511:0] xnor_vec;
            wire [9:0] pop_count;
            wire raw_out;

            assign xnor_vec = ~(l1_in ^ weights_l2[j]);
            assign pop_count = $countones(xnor_vec);
            assign raw_out = (pop_count >= thresh_l2[j]) ? 1'b1 : 1'b0;
            assign l2_out[j] = raw_out ^ invert_l2[j][0];
        end
    endgenerate

endmodule
