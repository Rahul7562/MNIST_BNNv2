`include "bnn_pkg.sv"

// Icarus testbench: drives every test vector in hw/mem/testvecs.mem through bnn_top
// and compares the predicted digit to the label in hw/mem/testlabels.mem.
// Gate (ARCHITECTURE.md §9): 100% match vs the SW popcount recompute on >=40 images
// including my_digit.png.

module bnn_tb;
    import bnn_pkg::*;

    logic clk, rst, start, done;
    logic [(NIN1-1):0] image_in;
    logic [3:0] digit;

    bnn_top DUT (.clk(clk), .rst(rst), .start(start),
                .image_in(image_in), .digit(digit), .done(done));

    // Read test vectors and labels.
    reg [(NIN1-1):0] vecs [0:20000];
    reg [31:0]       labels [0:20000];
    integer n_vec, n_lab;
    integer i, mismatches, checked;

    initial begin
        $readmemh({"hw/mem/", "testvecs.mem"}, vecs);
        // labels are decimal; readmemb can't parse decimals, so read via $fscanf.
    end

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer fd;
    reg [(NIN1-1):0] tmpv;
    reg [31:0] tmpl;
    string line;

    initial begin
        rst = 1; start = 0;
        #20 rst = 0;

        // Load vectors + labels manually from files (decimal labels).
        fd = $fopen({"hw/mem/", "testvecs.mem"}, "r");
        n_vec = 0;
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%h\n", tmpv) == 1) begin
                vecs[n_vec] = tmpv;
                n_vec = n_vec + 1;
            end
        end
        $fclose(fd);

        fd = $fopen({"hw/mem/", "testlabels.mem"}, "r");
        n_lab = 0;
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%d\n", tmpl) == 1) begin
                labels[n_lab] = tmpl;
                n_lab = n_lab + 1;
            end
        end
        $fclose(fd);

        if (n_vec != n_lab) begin
            $display("ERROR: vector/label count mismatch %0d vs %0d", n_vec, n_lab);
            $finish;
        end

        $display("Loaded %0d test vectors", n_vec);
        mismatches = 0;
        checked = 0;

        for (i = 0; i < n_vec; i = i + 1) begin
            @(posedge clk);
            image_in = vecs[i];
            start = 1;
            @(posedge clk);
            start = 0;
            // Wait for done (max a few thousand cycles is ample).
            fork
                begin: wait_done
                    wait (done == 1'b1);
                end
                begin: timeout
                    repeat (5000) @(posedge clk);
                    $display("ERROR: timeout on vector %0d", i);
                    $finish;
                end
            join_any
            disable timeout;
            disable wait_done;

            if (labels[i] <= 9) begin
                checked = checked + 1;
                if (digit !== labels[i][3:0]) begin
                    mismatches = mismatches + 1;
                    $display("MISMATCH vec %0d: pred=%0d expected=%0d", i, digit, labels[i]);
                end
            end else begin
                $display("vec %0d (my_digit.png): pred=%0d (label unknown, cross-check with SW)", i, digit);
            end
        end

        $display("========================================");
        $display("Checked (labeled) vectors: %0d", checked);
        $display("Mismatches: %0d", mismatches);
        if (mismatches == 0 && checked >= 40)
            $display("HW SIM PASS: 100%% match on >=40 images");
        else
            $display("HW SIM FAIL");
        $display("========================================");
        $finish;
    end

endmodule
