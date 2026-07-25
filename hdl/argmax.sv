`timescale 1ns / 1ps

module argmax #(
    parameter CLASSES = 10
)(
    input  logic        clk,
    input  logic        rst,
    input  logic        en,
    input  logic [15:0] scores [CLASSES-1:0],
    output logic [3:0]  pred,
    output logic        valid
);

    logic [15:0] max_score;
    logic [3:0]  max_idx;
    integer i;

    always_comb begin
        max_score = scores[0];
        max_idx = 0;
        for (i = 1; i < CLASSES; i = i + 1) begin
            if (scores[i] > max_score) begin
                max_score = scores[i];
                max_idx = i;
            end
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pred <= '0;
            valid <= 1'b0;
        end else if (en) begin
            pred <= max_idx;
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
