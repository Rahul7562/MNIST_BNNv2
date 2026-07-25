`timescale 1ns / 1ps

module maxpool2d #(
    parameter CHANNELS = 32,
    parameter IMG_WIDTH = 26
)(
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [CHANNELS-1:0] pixel_in,
    output logic [CHANNELS-1:0] pixel_out,
    output logic valid_out
);

    logic [CHANNELS-1:0] row_buf [IMG_WIDTH-1:0];
    logic [CHANNELS-1:0] prev_in;
    logic [CHANNELS-1:0] prev_row_val;
    logic [CHANNELS-1:0] prev_row_prev_val;

    logic [7:0] col_cnt;
    logic [7:0] row_cnt;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            integer i;
            for (i=0; i<IMG_WIDTH; i++) row_buf[i] <= '0;
            prev_in <= '0;
            prev_row_val <= '0;
            prev_row_prev_val <= '0;

            col_cnt <= '0;
            row_cnt <= '0;
            pixel_out <= '0;
            valid_out <= 1'b0;
        end else if (en) begin
            integer i;
            for (i=IMG_WIDTH-1; i>0; i--) row_buf[i] <= row_buf[i-1];
            row_buf[0] <= pixel_in;

            prev_in <= pixel_in;
            prev_row_val <= row_buf[IMG_WIDTH-1];
            prev_row_prev_val <= prev_row_val;

            if (col_cnt == IMG_WIDTH-1) begin
                col_cnt <= '0;
                if (row_cnt == IMG_WIDTH-1) row_cnt <= '0;
                else row_cnt <= row_cnt + 1;
            end else begin
                col_cnt <= col_cnt + 1;
            end

            if (col_cnt[0] == 1'b1 && row_cnt[0] == 1'b1) begin
                pixel_out <= pixel_in | prev_in | row_buf[IMG_WIDTH-1] | prev_row_prev_val;
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
