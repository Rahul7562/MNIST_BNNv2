`timescale 1ns / 1ps

module line_buffer_3x3 #(
    parameter IN_CHANNELS = 1,
    parameter IMG_WIDTH = 28
)(
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [IN_CHANNELS-1:0] pixel_in,
    output logic [IN_CHANNELS*9-1:0] window_out,
    output logic valid_out
);

    logic [IN_CHANNELS-1:0] row1 [IMG_WIDTH-1:0];
    logic [IN_CHANNELS-1:0] row2 [IMG_WIDTH-1:0];
    logic [IN_CHANNELS-1:0] win [2:0][2:0];

    // Pixel counter to know when valid 3x3 window is available
    // For no padding, first valid output occurs when we receive the 3rd pixel of the 3rd row.
    // Total pixels = 2 * IMG_WIDTH + 3 = 2*28 + 3 = 59.
    // Also, at the end of each row, the last 2 pixels are invalid because the window wraps around.

    logic [15:0] count;
    logic [7:0] col_count;

    integer i;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i=0; i<IMG_WIDTH; i++) begin
                row1[i] <= '0;
                row2[i] <= '0;
            end
            for (i=0; i<3; i++) begin
                win[i][0] <= '0; win[i][1] <= '0; win[i][2] <= '0;
            end
            valid_out <= 1'b0;
            count <= '0;
            col_count <= '0;
        end else if (en) begin
            for (i=IMG_WIDTH-1; i>0; i--) begin
                row1[i] <= row1[i-1];
                row2[i] <= row2[i-1];
            end
            row1[0] <= pixel_in;
            row2[0] <= row1[IMG_WIDTH-1];

            win[0][2] <= win[0][1]; win[0][1] <= win[0][0]; win[0][0] <= row2[IMG_WIDTH-1];
            win[1][2] <= win[1][1]; win[1][1] <= win[1][0]; win[1][0] <= row1[IMG_WIDTH-1];
            win[2][2] <= win[2][1]; win[2][1] <= win[2][0]; win[2][0] <= pixel_in;

            count <= count + 1;
            if (col_count == IMG_WIDTH - 1)
                col_count <= '0;
            else
                col_count <= col_count + 1;

            // Valid if we have buffered at least 2 full rows + 3 pixels
            // AND we are not in the first 2 columns (which would be wrapping around from prev row)
            if (count >= (2 * IMG_WIDTH + 2) && col_count >= 2) begin
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

    // We must match PyTorch layout for XNOR weights: In_Channels -> Height -> Width.
    // PyTorch flattened format puts W innermost, then H, then C (assuming N, C, H, W).
    // Actually PyTorch flattened tensor of [C, H, W] in memory is contiguous.
    // Index = c * (H*W) + h * W + w.
    // So for a 3x3 window over C channels, PyTorch flattens as:
    // C0_H0_W0, C0_H0_W1, C0_H0_W2, C0_H1_W0 ... C1_H0_W0 ...

    always_comb begin
        integer c, r, x;
        for (c=0; c<IN_CHANNELS; c++) begin
            for (r=0; r<3; r++) begin
                for (x=0; x<3; x++) begin
                    // PyTorch flattening: c * 9 + r * 3 + x
                    // win[r][x] has channels bitwise
                    window_out[c*9 + r*3 + x] = win[r][x][c];
                end
            end
        end
    end
endmodule
