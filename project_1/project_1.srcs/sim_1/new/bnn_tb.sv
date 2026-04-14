`timescale 1ns/1ps

`ifndef MEM_PATH
`define MEM_PATH "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/"
`endif

module bnn_tb;

    logic clk;
    logic rst_n;
    logic start;
    logic [783:0] image_in;
    logic [3:0] pred_digit;
    logic done;

    reg [783:0] test_images [0:9];
    reg [783:0] image_word [0:0];
    bit image_loaded [0:9];

    integer pass_count;
    integer fail_count;
    integer load_fail_count;
    integer i;

    bnn_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .image_in(image_in),
        .pred_digit(pred_digit),
        .done(done)
    );

    always #5 clk = ~clk;

    task automatic load_image(input int idx);
        string roots [0:3];
        string file_path;
        integer fd;
        int p;
        bit loaded;
        begin
            roots[0] = `MEM_PATH;
            roots[1] = "mem_files/";
            roots[2] = "../mem_files/";
            roots[3] = "../../mem_files/";
            loaded = 1'b0;

            for (p = 0; p < 4; p = p + 1) begin
                file_path = $sformatf("%stest_image_%0d.mem", roots[p], idx);
                fd = $fopen(file_path, "r");
                if (fd != 0) begin
                    $fclose(fd);
                    $readmemb(file_path, image_word);
                    test_images[idx] = image_word[0];
                    image_loaded[idx] = 1'b1;
                    loaded = 1'b1;
                    $display("Loaded image %0d from %s", idx, file_path);
                    break;
                end
            end

            if (!loaded) begin
                image_loaded[idx] = 1'b0;
                test_images[idx] = '0;
                load_fail_count = load_fail_count + 1;
                $error("Could not locate test_image_%0d.mem (tried absolute path and fallbacks)", idx);
            end
        end
    endtask

    task automatic run_case(input int expected_digit);
        int cycles;
        bit got_done;
        logic [3:0] expected_digit_4b;
        begin
            if (!image_loaded[expected_digit]) begin
                $error("Image %0d was not loaded, skipping inference", expected_digit);
                fail_count = fail_count + 1;
            end else begin
            expected_digit_4b = expected_digit[3:0];
            image_in = test_images[expected_digit];
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;

            got_done = 1'b0;
            for (cycles = 0; cycles < 20; cycles = cycles + 1) begin
                @(posedge clk);
                if (done == 1'b1) begin
                    got_done = 1'b1;
                    break;
                end
            end

            if (!got_done) begin
                $error("Image %0d: Timeout waiting for done", expected_digit);
                fail_count = fail_count + 1;
            end else begin
                $display("Image %0d: Predicted = %0d", expected_digit, pred_digit);
                if (pred_digit !== expected_digit_4b) begin
                    $error("Image %0d: Expected %0d, got %0d", expected_digit, expected_digit_4b, pred_digit);
                    fail_count = fail_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                end
                @(posedge clk);
            end
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        image_in = '0;
        pass_count = 0;
        fail_count = 0;
        load_fail_count = 0;

        for (i = 0; i < 10; i = i + 1) begin
            load_image(i);
        end

        if (load_fail_count != 0) begin
            $warning("%0d test image file(s) failed to load", load_fail_count);
        end

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        for (i = 0; i < 10; i = i + 1) begin
            run_case(i);
        end

        $display("BNN summary: pass=%0d fail=%0d", pass_count, fail_count);
        if (fail_count == 0) begin
            $display("BNN test PASSED");
        end else begin
            $fatal(1, "BNN test FAILED with %0d failing cases", fail_count);
        end

        $finish;
    end

endmodule
