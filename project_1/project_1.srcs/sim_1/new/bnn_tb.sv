`timescale 1ns/1ps

module bnn_tb;

    logic clk;
    logic rst;
    logic start;
    logic [783:0] input_digit;
    logic [3:0] predicted_digit;
    logic pass;
    logic fail;

    reg [783:0] input_mem [0:0];

    localparam logic [3:0] EXPECTED_DIGIT = 4'd5;

    bnn_top dut (
        .clk(clk),
        .rst_n(~rst),
        .start(start),
        .image_in(input_digit),
        .pred_digit(predicted_digit),
        .done()
    );

    always #5 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;
        input_digit = '0;
        pass = 1'b0;
        fail = 1'b0;

        $readmemb("input.mem", input_mem);
        input_digit = input_mem[0];

        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);

        $display("Predicted: %0d, Expected: %0d", predicted_digit, EXPECTED_DIGIT);
        if (predicted_digit == EXPECTED_DIGIT) begin
            pass = 1'b1;
            $display("PASS");
        end else begin
            fail = 1'b1;
            $display("FAIL");
        end

        $finish;
    end

endmodule
