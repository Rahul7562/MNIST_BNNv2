`timescale 1ns/1ps

module bnn_tb;

    logic clk;
    logic rst;
    logic start;
    logic [3:0] predicted_digit;
    logic done;
    logic pass;
    logic fail;
    integer cycle_cnt;
    localparam int MAX_WAIT_CYCLES = 10000000;

    localparam logic [3:0] EXPECTED_DIGIT = 4'd5;

    bnn_top dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .led(predicted_digit),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;
        pass = 1'b0;
        fail = 1'b0;

        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        rst = 1'b0;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        // Sequential design now takes many cycles; wait for completion pulse with timeout.
        cycle_cnt = 0;
        while ((done == 1'b0) && (cycle_cnt < MAX_WAIT_CYCLES)) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
        end
        if (done == 1'b0) begin
            fail = 1'b1;
            $display("FAIL: timeout waiting for done");
            $finish;
        end
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
