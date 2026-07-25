`timescale 1ns / 1ps

module flatten #(
    parameter CHANNELS = 64,
    parameter IMG_WIDTH = 3
)(
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [CHANNELS-1:0] pixel_in,
    output logic [CHANNELS*IMG_WIDTH*IMG_WIDTH-1:0] feat_out,
    output logic valid_out
);
    localparam TOTAL_PIXELS = IMG_WIDTH * IMG_WIDTH;

    // Store incoming pixels
    logic [CHANNELS-1:0] shift_reg [TOTAL_PIXELS-1:0];
    logic [7:0] cnt;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            integer i;
            for (i=0; i<TOTAL_PIXELS; i++) shift_reg[i] <= '0;
            cnt <= '0;
            valid_out <= 1'b0;
        end else if (en) begin
            integer i;
            for (i=TOTAL_PIXELS-1; i>0; i--) shift_reg[i] <= shift_reg[i-1];
            shift_reg[0] <= pixel_in;

            if (cnt == TOTAL_PIXELS - 1) begin
                cnt <= '0;
                valid_out <= 1'b1;
            end else begin
                cnt <= cnt + 1;
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

    // Convert from pixel-stream to PyTorch's (C, H, W) flatten format.
    // shift_reg contains the pixels.
    // shift_reg[TOTAL_PIXELS-1] is the TOP-LEFT pixel (first one received).
    // shift_reg[0] is the BOTTOM-RIGHT pixel (last one received).
    // So index `p` from 0 to TOTAL_PIXELS-1 mapping to image:
    // Pixel index `p` (where 0 is top-left) is `shift_reg[TOTAL_PIXELS-1-p]`.
    // PyTorch flattens (C, H, W) such that C is the outermost loop.
    // Flattened array = [ C0_P0, C0_P1, ..., C0_P(N-1), C1_P0, ... ]

    always_comb begin
        integer c, p;
        for (c=0; c<CHANNELS; c++) begin
            for (p=0; p<TOTAL_PIXELS; p++) begin
                // Flattened index = c * TOTAL_PIXELS + p
                feat_out[c*TOTAL_PIXELS + p] = shift_reg[TOTAL_PIXELS-1-p][c];
            end
        end
    end

endmodule
