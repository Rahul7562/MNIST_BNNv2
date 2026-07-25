`timescale 1ns / 1ps

module bnn_tb();

    logic clk;
    logic rst;
    logic start;
    logic image_en;
    logic image_pixel_in;
    logic [3:0] led;
    logic done;

    bnn_top dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .image_en(image_en),
        .image_pixel_in(image_pixel_in),
        .led(led),
        .done(done)
    );

    always #5 clk = ~clk;

    logic image_data [0:783]; // Fixed index to load array sequentially
    integer i, fd;

    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        image_en = 0;
        image_pixel_in = 0;

        #20 rst = 0;
        #10;

        // Load image from file (Python output)
        $readmemb("tests/data/img_0.mem", image_data);

        start = 1;
        #10 start = 0;

        // Stream image
        for (i=0; i<784; i++) begin
            image_en = 1;
            image_pixel_in = image_data[i];
            #10;
        end
        image_en = 0;

        // Wait for done
        wait(done == 1'b1);
        #10;
        $display("Inference Complete. Prediction: %d", led);

        // $finish; // uncomment for actual sim
    end

endmodule
