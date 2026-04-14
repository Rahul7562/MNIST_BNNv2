`ifndef MEM_PATH
`define MEM_PATH "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/"
`endif

module bnn_layer1 (
    input  logic [783:0] image_in,
    output logic [511:0] l1_out
);

    reg [783:0] weights_l1 [0:511];
    reg [9:0]   thresh_l1  [0:511];
    reg [0:0]   invert_l1  [0:511];

    initial begin
        $readmemb({`MEM_PATH, "weights_l1.mem"}, weights_l1);
        $readmemb({`MEM_PATH, "thresh_l1.mem"},  thresh_l1);
        $readmemb({`MEM_PATH, "invert_l1.mem"},  invert_l1);
    end

    genvar j;
    generate
        for (j = 0; j < 512; j = j + 1) begin : g_l1_neuron
            wire [783:0] xnor_vec;
            wire [9:0] pop_count;
            wire raw_out;

            assign xnor_vec = ~(image_in ^ weights_l1[j]);
            assign pop_count = $countones(xnor_vec);
            assign raw_out = (pop_count >= thresh_l1[j]) ? 1'b1 : 1'b0;
            assign l1_out[j] = raw_out ^ invert_l1[j][0];
        end
    endgenerate

endmodule
