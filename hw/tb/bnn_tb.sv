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
    localparam int MAX_WAIT_CYCLES = 100000;

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
        #1; rst = 1'b0;
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);

        #1; start = 1'b1;
        @(posedge clk);
        #1; start = 1'b0;

        cycle_cnt = 0;
        while ((done == 1'b0) && (cycle_cnt < MAX_WAIT_CYCLES)) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
        end
        if (done == 1'b0) begin
            fail = 1'b1;
            $display("FAIL: timeout waiting for done, state=%0d", dut.state);
            $finish;
        end
        @(posedge clk);

        $display("Predicted: %0d, Cycles: %0d", predicted_digit, cycle_cnt);
        $finish;
    end

endmodule
