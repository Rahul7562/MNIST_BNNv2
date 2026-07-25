`timescale 1ns / 1ps

module bnn_top (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic        image_en,
    input  logic [0:0]  image_pixel_in,
    output logic [3:0]  led,
    output logic        done
);

    logic [31:0] l1_out; logic l1_valid;
    conv2d_bnn #(
        .IN_CHANNELS(1), .OUT_CHANNELS(32), .IMG_WIDTH(28),
        .WEIGHT_FILE("mem_files/conv1_weights.mem"), .THRESH_FILE("mem_files/bn1_thresh.mem")
    ) layer1 (
        .clk(clk), .rst(rst), .en(image_en), .pixel_in(image_pixel_in), .pixel_out(l1_out), .valid_out(l1_valid)
    );

    logic [31:0] p1_out; logic p1_valid;
    maxpool2d #(
        .CHANNELS(32), .IMG_WIDTH(26)
    ) pool1 (
        .clk(clk), .rst(rst), .en(l1_valid), .pixel_in(l1_out), .pixel_out(p1_out), .valid_out(p1_valid)
    );

    logic [63:0] l2_out; logic l2_valid;
    conv2d_bnn #(
        .IN_CHANNELS(32), .OUT_CHANNELS(64), .IMG_WIDTH(13),
        .WEIGHT_FILE("mem_files/conv2_weights.mem"), .THRESH_FILE("mem_files/bn2_thresh.mem")
    ) layer2 (
        .clk(clk), .rst(rst), .en(p1_valid), .pixel_in(p1_out), .pixel_out(l2_out), .valid_out(l2_valid)
    );

    logic [63:0] p2_out; logic p2_valid;
    maxpool2d #(
        .CHANNELS(64), .IMG_WIDTH(11)
    ) pool2 (
        .clk(clk), .rst(rst), .en(l2_valid), .pixel_in(l2_out), .pixel_out(p2_out), .valid_out(p2_valid)
    );

    logic [63:0] l3_out; logic l3_valid;
    conv2d_bnn #(
        .IN_CHANNELS(64), .OUT_CHANNELS(64), .IMG_WIDTH(5),
        .WEIGHT_FILE("mem_files/conv3_weights.mem"), .THRESH_FILE("mem_files/bn3_thresh.mem")
    ) layer3 (
        .clk(clk), .rst(rst), .en(p2_valid), .pixel_in(p2_out), .pixel_out(l3_out), .valid_out(l3_valid)
    );

    logic [64*3*3-1:0] flat_out; logic flat_valid;
    flatten #(
        .CHANNELS(64), .IMG_WIDTH(3)
    ) flatten_layer (
        .clk(clk), .rst(rst), .en(l3_valid), .pixel_in(l3_out), .feat_out(flat_out), .valid_out(flat_valid)
    );

    logic [63:0] fc1_out; logic fc1_valid;
    dense_layer #(
        .IN_FEATURES(64*3*3), .OUT_FEATURES(64),
        .WEIGHT_FILE("mem_files/fc1_weights.mem"), .THRESH_FILE("mem_files/bn4_thresh.mem"), .IS_OUTPUT_LAYER(0)
    ) fc1 (
        .clk(clk), .rst(rst), .en(flat_valid), .feat_in(flat_out), .feat_out(fc1_out), .valid_out(fc1_valid)
    );

    logic [15:0] fc2_scores [9:0]; logic fc2_valid;
    dense_layer #(
        .IN_FEATURES(64), .OUT_FEATURES(10),
        .WEIGHT_FILE("mem_files/fc2_weights.mem"), .THRESH_FILE("mem_files/bn5_thresh.mem"), .IS_OUTPUT_LAYER(1)
    ) fc2 (
        .clk(clk), .rst(rst), .en(fc1_valid), .feat_in(fc1_out), .scores_out(fc2_scores), .valid_out(fc2_valid)
    );

    argmax #(.CLASSES(10)) argmax_layer (
        .clk(clk), .rst(rst), .en(fc2_valid), .scores(fc2_scores), .pred(led), .valid(done)
    );

endmodule
