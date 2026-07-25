`timescale 1ns / 1ps

module mac_xnor_popcount #(
    parameter N = 9
)(
    input  logic         clk,
    input  logic         rst,
    input  logic         en,
    input  logic [N-1:0] act,
    input  logic [N-1:0] w,
    output logic [15:0]  popcount,
    output logic         valid
);

    logic [N-1:0] xnor_res;
    assign xnor_res = ~(act ^ w);

    logic [15:0] count;
    integer i;
    always_comb begin
        count = 0;
        for (i = 0; i < N; i = i + 1) begin
            count = count + xnor_res[i];
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            popcount <= '0;
            valid <= 1'b0;
        end else if (en) begin
            popcount <= count;
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
