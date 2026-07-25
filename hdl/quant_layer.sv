`timescale 1ns / 1ps

module quant_layer #(
    parameter N = 9
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,
    input  logic [15:0] popcount_in,
    input  logic [16:0] thresh_in,
    output logic        act_out,
    output logic        valid_out
);

    logic invert;
    logic [15:0] threshold;
    assign invert = thresh_in[16];
    assign threshold = thresh_in[15:0];

    logic cmp;
    assign cmp = (popcount_in >= threshold);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            act_out <= 1'b0;
            valid_out <= 1'b0;
        end else if (en) begin
            act_out <= invert ? ~cmp : cmp;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule
