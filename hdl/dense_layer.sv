`timescale 1ns / 1ps

module dense_layer #(
    parameter IN_FEATURES = 64 * 3 * 3,
    parameter OUT_FEATURES = 64,
    parameter WEIGHT_FILE = "",
    parameter THRESH_FILE = "",
    parameter IS_OUTPUT_LAYER = 0
)(
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [IN_FEATURES-1:0] feat_in,
    output logic [OUT_FEATURES-1:0] feat_out,
    output logic [15:0] scores_out [OUT_FEATURES-1:0],
    output logic valid_out
);

    logic [IN_FEATURES-1:0] weights [OUT_FEATURES-1:0];
    logic [16:0] thresholds [OUT_FEATURES-1:0];

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (THRESH_FILE != "") $readmemh(THRESH_FILE, thresholds);
    end

    logic [OUT_FEATURES-1:0] v_out;

    genvar i;
    generate
        for (i=0; i<OUT_FEATURES; i++) begin : macs
            logic [15:0] popcnt;
            logic mac_valid;

            mac_xnor_popcount #(.N(IN_FEATURES)) mac (
                .clk(clk),
                .rst(rst),
                .en(en),
                .act(feat_in),
                .w(weights[i]),
                .popcount(popcnt),
                .valid(mac_valid)
            );

            if (IS_OUTPUT_LAYER) begin
                assign scores_out[i] = popcnt;
                always_ff @(posedge clk) begin
                    if (rst) v_out[i] <= 0;
                    else v_out[i] <= mac_valid;
                end
            end else begin
                quant_layer #(.N(IN_FEATURES)) quant (
                    .clk(clk),
                    .rst(rst),
                    .en(mac_valid),
                    .popcount_in(popcnt),
                    .thresh_in(thresholds[i]),
                    .act_out(feat_out[i]),
                    .valid_out(v_out[i])
                );
            end
        end
    endgenerate

    assign valid_out = v_out[0];

endmodule
