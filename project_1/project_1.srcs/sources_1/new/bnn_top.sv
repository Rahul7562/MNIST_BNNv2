`ifndef MEM_PATH
`define MEM_PATH "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/"
`endif

module bnn_top (
    input clk,                     // FIX: minimal external IO for FPGA top-level.
    input rst,                     // FIX: active-high reset input for board button mapping.
    input start,
    output [3:0] led,              // FIX: expose 4-bit prediction on LEDs.
    output done
);

    localparam int CHUNK_BITS  = 16;
    localparam int L1_LAST_BIT = 784 - CHUNK_BITS;
    localparam int L2_LAST_BIT = 512 - CHUNK_BITS;

    typedef enum logic [3:0] {
        IDLE,
        L1_FETCH,   // FIX: register L1 chunk operands before popcount.
        L1_POP,     // FIX: register L1 popcount in dedicated stage.
        L1_ACCUM,
        L1_WRITE,
        L2_FETCH,   // FIX: register L2 chunk operands before popcount.
        L2_POP,     // FIX: register L2 popcount in dedicated stage.
        L2_ACCUM,
        L2_WRITE,
        OUT_CLASS_INIT,
        OUT_ACCUM,
        OUT_STORE,
        ARGMAX_INIT,
        ARGMAX_LOOP,
        DONE
    } state_t;

    state_t state;

    logic [783:0] image_reg;
    // FIX: removed external 784-bit input bus and replaced it with internal image memory.
    reg   [783:0] image_mem [0:0];
    logic [511:0] l1_reg;
    logic [255:0] l2_reg;
    logic [3:0] pred_reg;
    logic [3:0] pred_digit; // FIX: internal prediction bus mapped to LED outputs.

    // Model memories kept in top and accessed by index to avoid massive parallel logic.
    reg [783:0] weights_l1 [0:511];
    reg [9:0]   thresh_l1  [0:511];
    reg [0:0]   invert_l1  [0:511];

    reg [511:0] weights_l2 [0:255];
    reg [9:0]   thresh_l2  [0:255];
    reg [0:0]   invert_l2  [0:255];

    reg signed [15:0] weights_out [0:9][0:255];
    reg signed [15:0] bias_out [0:9];
    reg [15:0] weights_out_flat [0:2559];

    logic [8:0] l1_neuron_idx;
    logic [7:0] l2_neuron_idx;
    logic [9:0] bit_idx;
    logic [3:0] class_idx;
    logic [7:0] feat_idx;
    logic [10:0] pop_acc;

    // FIX: staged registers to break RAM->popcount->accumulator critical path.
    logic [CHUNK_BITS-1:0] l1_img_chunk_reg;
    logic [CHUNK_BITS-1:0] l1_w_chunk_reg;
    logic [4:0]            l1_pop_reg;
    logic                  l1_last_chunk_reg;

    logic [CHUNK_BITS-1:0] l2_feat_chunk_reg;
    logic [CHUNK_BITS-1:0] l2_w_chunk_reg;
    logic [4:0]            l2_pop_reg;
    logic                  l2_last_chunk_reg;

    logic signed [31:0] out_acc;
    logic signed [31:0] class_scores [0:9];
    logic signed [31:0] best_val;
    logic [3:0] best_idx;

    integer k;
    integer j;

    function automatic [4:0] popcount16(input logic [15:0] vec);
        integer i;
        begin
            popcount16 = 5'd0;
            for (i = 0; i < 16; i = i + 1) begin
                popcount16 = popcount16 + vec[i];
            end
        end
    endfunction

    initial begin
        // FIX: image is now loaded internally from memory file to reduce top-level IO usage.
        $readmemb({`MEM_PATH, "input.mem"}, image_mem);

        $readmemb({`MEM_PATH, "weights_l1.mem"}, weights_l1);
        $readmemb({`MEM_PATH, "thresh_l1.mem"},  thresh_l1);
        $readmemb({`MEM_PATH, "invert_l1.mem"},  invert_l1);

        $readmemb({`MEM_PATH, "weights_l2.mem"}, weights_l2);
        $readmemb({`MEM_PATH, "thresh_l2.mem"},  thresh_l2);
        $readmemb({`MEM_PATH, "invert_l2.mem"},  invert_l2);

        $readmemh({`MEM_PATH, "weights_out.mem"}, weights_out_flat);
        for (k = 0; k < 10; k = k + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                weights_out[k][j] = $signed(weights_out_flat[(k * 256) + j]);
            end
        end
        $readmemh({`MEM_PATH, "bias_out.mem"}, bias_out);
    end

    always_ff @(posedge clk) begin // FIX: use synchronous reset style for FPGA-friendly timing.
        if (rst) begin
            state          <= IDLE;
            image_reg      <= '0;
            l1_reg         <= '0;
            l2_reg         <= '0;
            pred_reg       <= '0;
            l1_neuron_idx  <= '0;
            l2_neuron_idx  <= '0;
            bit_idx        <= '0;
            class_idx      <= '0;
            feat_idx       <= '0;
            pop_acc        <= '0;
            l1_img_chunk_reg  <= '0;
            l1_w_chunk_reg    <= '0;
            l1_pop_reg        <= '0;
            l1_last_chunk_reg <= 1'b0;
            l2_feat_chunk_reg  <= '0;
            l2_w_chunk_reg     <= '0;
            l2_pop_reg         <= '0;
            l2_last_chunk_reg  <= 1'b0;
            out_acc        <= '0;
            best_val       <= '0;
            best_idx       <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        // FIX: consume internal memory image instead of external bus.
                        image_reg     <= image_mem[0];
                        l1_reg        <= '0;
                        l2_reg        <= '0;
                        pred_reg      <= '0;
                        l1_neuron_idx <= 9'd0;
                        bit_idx       <= 10'd0;
                        pop_acc       <= 11'd0;
                        state         <= L1_FETCH;
                    end
                end

                // FIX: Stage 1 - register operand chunks from memory/source vector.
                L1_FETCH: begin
                    l1_img_chunk_reg  <= image_reg[bit_idx +: CHUNK_BITS];
                    l1_w_chunk_reg    <= weights_l1[l1_neuron_idx][bit_idx +: CHUNK_BITS];
                    l1_last_chunk_reg <= (bit_idx == L1_LAST_BIT[9:0]);
                    state <= L1_POP;
                end

                // FIX: Stage 2 - popcount from registered XNOR chunk.
                L1_POP: begin
                    l1_pop_reg <= popcount16(~(l1_img_chunk_reg ^ l1_w_chunk_reg));
                    state <= L1_ACCUM;
                end

                // FIX: Stage 3 - accumulate registered popcount.
                L1_ACCUM: begin
                    pop_acc <= pop_acc + l1_pop_reg;

                    if (l1_last_chunk_reg) begin
                        state <= L1_WRITE;
                    end else begin
                        bit_idx <= bit_idx + CHUNK_BITS;
                        state <= L1_FETCH;
                    end
                end

                // Stage 4: threshold/activation and move to next neuron.
                L1_WRITE: begin
                    l1_reg[l1_neuron_idx] <=
                        ((pop_acc >= thresh_l1[l1_neuron_idx]) ? 1'b1 : 1'b0) ^ invert_l1[l1_neuron_idx][0];

                    if (l1_neuron_idx == 9'd511) begin
                        l2_neuron_idx <= 8'd0;
                        bit_idx       <= 10'd0;
                        pop_acc       <= 11'd0;
                        state         <= L2_FETCH;
                    end else begin
                        l1_neuron_idx <= l1_neuron_idx + 9'd1;
                        bit_idx       <= 10'd0;
                        pop_acc       <= 11'd0;
                        state         <= L1_FETCH;
                    end
                end

                // FIX: Stage 1 - register operand chunks from memory/source vector.
                L2_FETCH: begin
                    l2_feat_chunk_reg  <= l1_reg[bit_idx +: CHUNK_BITS];
                    l2_w_chunk_reg     <= weights_l2[l2_neuron_idx][bit_idx +: CHUNK_BITS];
                    l2_last_chunk_reg  <= (bit_idx == L2_LAST_BIT[9:0]);
                    state <= L2_POP;
                end

                // FIX: Stage 2 - popcount from registered XNOR chunk.
                L2_POP: begin
                    l2_pop_reg <= popcount16(~(l2_feat_chunk_reg ^ l2_w_chunk_reg));
                    state <= L2_ACCUM;
                end

                L2_ACCUM: begin
                    pop_acc <= pop_acc + l2_pop_reg;

                    if (l2_last_chunk_reg) begin
                        state <= L2_WRITE;
                    end else begin
                        bit_idx <= bit_idx + CHUNK_BITS;
                        state <= L2_FETCH;
                    end
                end

                L2_WRITE: begin
                    l2_reg[l2_neuron_idx] <=
                        ((pop_acc >= thresh_l2[l2_neuron_idx]) ? 1'b1 : 1'b0) ^ invert_l2[l2_neuron_idx][0];

                    if (l2_neuron_idx == 8'd255) begin
                        class_idx <= 4'd0;
                        state <= OUT_CLASS_INIT;
                    end else begin
                        l2_neuron_idx <= l2_neuron_idx + 8'd1;
                        bit_idx <= 10'd0;
                        pop_acc <= 11'd0;
                        state <= L2_FETCH;
                    end
                end

                OUT_CLASS_INIT: begin
                    feat_idx <= 8'd0;
                    out_acc  <= $signed(bias_out[class_idx]);
                    state    <= OUT_ACCUM;
                end

                // Output layer is also time-multiplexed: one feature contribution per cycle.
                OUT_ACCUM: begin
                    if (l2_reg[feat_idx]) begin
                        out_acc <= out_acc + $signed(weights_out[class_idx][feat_idx]);
                    end else begin
                        out_acc <= out_acc - $signed(weights_out[class_idx][feat_idx]);
                    end

                    if (feat_idx == 8'd255) begin
                        state <= OUT_STORE;
                    end else begin
                        feat_idx <= feat_idx + 8'd1;
                    end
                end

                OUT_STORE: begin
                    class_scores[class_idx] <= out_acc;

                    if (class_idx == 4'd9) begin
                        state <= ARGMAX_INIT;
                    end else begin
                        class_idx <= class_idx + 4'd1;
                        state <= OUT_CLASS_INIT;
                    end
                end

                ARGMAX_INIT: begin
                    best_val  <= class_scores[0];
                    best_idx  <= 4'd0;
                    class_idx <= 4'd1;
                    state     <= ARGMAX_LOOP;
                end

                ARGMAX_LOOP: begin
                    if (class_scores[class_idx] > best_val) begin
                        best_val <= class_scores[class_idx];
                        best_idx <= class_idx;
                    end

                    if (class_idx == 4'd9) begin
                        if (class_scores[class_idx] > best_val) begin
                            pred_reg <= class_idx;
                        end else begin
                            pred_reg <= best_idx;
                        end
                        state <= DONE;
                    end else begin
                        class_idx <= class_idx + 4'd1;
                    end
                end

                DONE: begin
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    assign pred_digit = pred_reg;
    assign led = pred_digit; // FIX: map predicted digit directly to LED outputs.
    assign done = (state == DONE);

endmodule
