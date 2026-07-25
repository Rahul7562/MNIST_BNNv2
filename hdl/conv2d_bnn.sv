`timescale 1ns / 1ps

module conv2d_bnn #(
    parameter IN_CHANNELS = 1,
    parameter OUT_CHANNELS = 32,
    parameter IMG_WIDTH = 28,
    parameter WEIGHT_FILE = "",
    parameter THRESH_FILE = ""
)(
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [IN_CHANNELS-1:0] pixel_in,
    output logic [OUT_CHANNELS-1:0] pixel_out,
    output logic valid_out
);
    localparam WINDOW_BITS = IN_CHANNELS * 9;

    logic [WINDOW_BITS-1:0] window;
    logic window_valid;

    line_buffer_3x3 #(
        .IN_CHANNELS(IN_CHANNELS),
        .IMG_WIDTH(IMG_WIDTH)
    ) lb (
        .clk(clk),
        .rst(rst),
        .en(en),
        .pixel_in(pixel_in),
        .window_out(window),
        .valid_out(window_valid)
    );

    logic [WINDOW_BITS-1:0] weights [OUT_CHANNELS-1:0];
    logic [16:0] thresholds [OUT_CHANNELS-1:0];

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weights);
        if (THRESH_FILE != "") $readmemh(THRESH_FILE, thresholds);
    end

    logic [OUT_CHANNELS-1:0] v_out;

    genvar i;
    generate
        for (i=0; i<OUT_CHANNELS; i++) begin : macs
            logic [15:0] popcnt;
            logic mac_valid;

            mac_xnor_popcount #(.N(WINDOW_BITS)) mac (
                .clk(clk),
                .rst(rst),
                .en(window_valid),
                .act(window),
                .w(weights[i]),
                .popcount(popcnt),
                .valid(mac_valid)
            );

            quant_layer #(.N(WINDOW_BITS)) quant (
                .clk(clk),
                .rst(rst),
                .en(mac_valid),
                .popcount_in(popcnt),
                .thresh_in(thresholds[i]),
                .act_out(pixel_out[i]),
                .valid_out(v_out[i])
            );
        end
    endgenerate

    assign valid_out = v_out[0];

endmodule
