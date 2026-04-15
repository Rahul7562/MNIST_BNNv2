`timescale 1ns/1ps

module bnn_tb;

    logic        clk;
    logic        rst;
    logic        start;
    logic [783:0] input_digit;
    logic [3:0]  predicted_digit;
    logic done;

    reg [783:0] input_mem [0:0];

    bnn_top dut (
        .clk        (clk),
        .rst_n      (~rst),
        .start      (start),
        .image_in   (input_digit),
        .pred_digit (predicted_digit),
        .done       (done)
    );

    always #5 clk = ~clk;

    initial begin
        clk         = 1'b0;
        rst         = 1'b1;
        start       = 1'b0;
        input_digit = '0;

        $readmemb("input.mem", input_mem);
        input_digit = input_mem[0];

        
        @(negedge clk); rst = 1'b0;

        @(posedge clk); start = 1'b1;
        @(posedge clk); start = 1'b0;

        @(posedge done);

        $display("Predicted: %0d", predicted_digit);

        repeat (5) @(posedge clk);
        $finish;
    end

endmodule