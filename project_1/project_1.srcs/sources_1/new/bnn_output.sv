`ifndef MEM_PATH
`define MEM_PATH "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/"
`endif

module bnn_output (
    input  logic [255:0] l2_in,
    output logic [3:0]   pred_digit
);

    reg signed [15:0] weights_out [0:9][0:255];
    reg signed [15:0] bias_out [0:9];
    reg [15:0] weights_out_flat [0:2559];

    logic signed [31:0] acc [0:9];
    logic signed [31:0] best_val;
    logic [3:0] best_idx;
    integer k;
    integer j;

    initial begin
        $readmemh({`MEM_PATH, "weights_out.mem"}, weights_out_flat);
        for (k = 0; k < 10; k = k + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                weights_out[k][j] = $signed(weights_out_flat[(k * 256) + j]);
            end
        end
        $readmemh({`MEM_PATH, "bias_out.mem"}, bias_out);
    end

    always_comb begin
        for (k = 0; k < 10; k = k + 1) begin
            acc[k] = $signed(bias_out[k]);
            for (j = 0; j < 256; j = j + 1) begin
                if (l2_in[j] == 1'b1) begin
                    acc[k] = acc[k] + $signed(weights_out[k][j]);
                end else begin
                    acc[k] = acc[k] - $signed(weights_out[k][j]);
                end
            end
        end

        best_idx = 4'd0;
        best_val = acc[0];
        for (k = 1; k < 10; k = k + 1) begin
            if (acc[k] > best_val) begin
                best_val = acc[k];
                best_idx = k[3:0];
            end
        end

        pred_digit = best_idx;
    end

endmodule
