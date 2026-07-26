`ifndef MEM_PATH
`define MEM_PATH "../mem_files/"
`endif

module bnn_top (
    input clk,
    input rst,
    input start,
    output [3:0] led,
    output done
);

    localparam int CHUNK_BITS  = 16;
    localparam int L1_MAX = 783;
    localparam int L2_MAX = 255;
    localparam int L3_MAX = 255;

    localparam int L1_LAST_BIT = L1_MAX + 1 - CHUNK_BITS;
    localparam int L2_LAST_BIT = L2_MAX + 1 - CHUNK_BITS;
    localparam int L3_LAST_BIT = L3_MAX + 1 - CHUNK_BITS;

    typedef enum logic [4:0] {
        IDLE,
        L1_FETCH,
        L1_POP,
        L1_ACCUM,
        L1_WRITE,
        L2_FETCH,
        L2_POP,
        L2_ACCUM,
        L2_WRITE,
        L3_FETCH,
        L3_POP,
        L3_ACCUM,
        L3_WRITE,
        ARGMAX_INIT,
        ARGMAX_LOOP,
        DONE
    } state_t;

    state_t state;

    logic [783:0] image_reg;
    reg   [783:0] image_mem [0:0];
    logic [255:0] l1_reg;
    logic [255:0] l2_reg;
    logic [3:0] pred_reg;
    logic [3:0] pred_digit;

    // Layer 1
    reg [783:0] weights_l1 [0:255];
    reg [15:0]  thresh_l1  [0:255];

    // Layer 2
    reg [255:0] weights_l2 [0:255];
    reg [15:0]  thresh_l2  [0:255];

    // Layer 3 (Output)
    reg [255:0] weights_l3 [0:9];
    reg signed [31:0] thresh_l3  [0:9];

    logic [8:0] l1_neuron_idx;
    logic [8:0] l2_neuron_idx;
    logic [3:0] class_idx;
    logic [9:0] bit_idx;
    logic [10:0] pop_acc;

    logic [CHUNK_BITS-1:0] l1_img_chunk_reg;
    logic [CHUNK_BITS-1:0] l1_w_chunk_reg;
    logic [4:0]            l1_pop_reg;
    logic                  l1_last_chunk_reg;

    logic [CHUNK_BITS-1:0] l2_feat_chunk_reg;
    logic [CHUNK_BITS-1:0] l2_w_chunk_reg;
    logic [4:0]            l2_pop_reg;
    logic                  l2_last_chunk_reg;

    logic [CHUNK_BITS-1:0] l3_feat_chunk_reg;
    logic [CHUNK_BITS-1:0] l3_w_chunk_reg;
    logic [4:0]            l3_pop_reg;
    logic                  l3_last_chunk_reg;

    logic signed [31:0] class_scores [0:9];
    logic signed [31:0] best_val;
    logic [3:0] best_idx;

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
        $readmemb({`MEM_PATH, "input.mem"}, image_mem);

        $readmemh({`MEM_PATH, "layer1_weights.mem"}, weights_l1);
        $readmemh({`MEM_PATH, "layer1_thresholds_int.mem"}, thresh_l1);

        $readmemh({`MEM_PATH, "layer2_weights.mem"}, weights_l2);
        $readmemh({`MEM_PATH, "layer2_thresholds_int.mem"}, thresh_l2);

        $readmemh({`MEM_PATH, "layer3_weights.mem"}, weights_l3);
        $readmemh({`MEM_PATH, "layer3_thresholds_int.mem"}, thresh_l3);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state          <= IDLE;
            image_reg      <= '0;
            l1_reg         <= '0;
            l2_reg         <= '0;
            pred_reg       <= '0;
            l1_neuron_idx  <= '0;
            l2_neuron_idx  <= '0;
            class_idx      <= '0;
            bit_idx        <= '0;
            pop_acc        <= '0;

            l1_img_chunk_reg  <= '0;
            l1_w_chunk_reg    <= '0;
            l1_pop_reg        <= '0;
            l1_last_chunk_reg <= 1'b0;

            l2_feat_chunk_reg <= '0;
            l2_w_chunk_reg    <= '0;
            l2_pop_reg        <= '0;
            l2_last_chunk_reg <= 1'b0;

            l3_feat_chunk_reg <= '0;
            l3_w_chunk_reg    <= '0;
            l3_pop_reg        <= '0;
            l3_last_chunk_reg <= 1'b0;

            best_val       <= '0;
            best_idx       <= '0;

            for (integer i=0; i<10; i++) class_scores[i] <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
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

                // Layer 1
                L1_FETCH: begin
                    l1_img_chunk_reg  <= image_reg[L1_MAX - bit_idx -: CHUNK_BITS];
                    l1_w_chunk_reg    <= weights_l1[l1_neuron_idx][L1_MAX - bit_idx -: CHUNK_BITS];
                    l1_last_chunk_reg <= (bit_idx == L1_LAST_BIT[9:0]);
                    state <= L1_POP;
                end

                L1_POP: begin
                    l1_pop_reg <= popcount16(~(l1_img_chunk_reg ^ l1_w_chunk_reg));
                    state <= L1_ACCUM;
                end

                L1_ACCUM: begin
                    pop_acc <= pop_acc + l1_pop_reg;

                    if (l1_last_chunk_reg) begin
                        state <= L1_WRITE;
                    end else begin
                        bit_idx <= bit_idx + CHUNK_BITS;
                        state <= L1_FETCH;
                    end
                end

                L1_WRITE: begin
                    l1_reg[L2_MAX - l1_neuron_idx] <= (pop_acc >= thresh_l1[l1_neuron_idx]) ? 1'b1 : 1'b0;

                    if (l1_neuron_idx == 9'd255) begin
                        l2_neuron_idx <= 9'd0;
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

                // Layer 2
                L2_FETCH: begin
                    l2_feat_chunk_reg  <= l1_reg[L2_MAX - bit_idx -: CHUNK_BITS];
                    l2_w_chunk_reg     <= weights_l2[l2_neuron_idx][L2_MAX - bit_idx -: CHUNK_BITS];
                    l2_last_chunk_reg  <= (bit_idx == L2_LAST_BIT[9:0]);
                    state <= L2_POP;
                end

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
                    l2_reg[L3_MAX - l2_neuron_idx] <= (pop_acc >= thresh_l2[l2_neuron_idx]) ? 1'b1 : 1'b0;

                    if (l2_neuron_idx == 9'd255) begin
                        class_idx <= 4'd0;
                        bit_idx   <= 10'd0;
                        pop_acc   <= 11'd0;
                        state     <= L3_FETCH;
                    end else begin
                        l2_neuron_idx <= l2_neuron_idx + 9'd1;
                        bit_idx       <= 10'd0;
                        pop_acc       <= 11'd0;
                        state         <= L2_FETCH;
                    end
                end

                // Layer 3
                L3_FETCH: begin
                    l3_feat_chunk_reg  <= l2_reg[L3_MAX - bit_idx -: CHUNK_BITS];
                    l3_w_chunk_reg     <= weights_l3[class_idx][L3_MAX - bit_idx -: CHUNK_BITS];
                    l3_last_chunk_reg  <= (bit_idx == L3_LAST_BIT[9:0]);
                    state <= L3_POP;
                end

                L3_POP: begin
                    l3_pop_reg <= popcount16(~(l3_feat_chunk_reg ^ l3_w_chunk_reg));
                    state <= L3_ACCUM;
                end

                L3_ACCUM: begin
                    pop_acc <= pop_acc + l3_pop_reg;

                    if (l3_last_chunk_reg) begin
                        state <= L3_WRITE;
                    end else begin
                        bit_idx <= bit_idx + CHUNK_BITS;
                        state <= L3_FETCH;
                    end
                end

                L3_WRITE: begin
                    // score = popcount * 256 - int_threshold
                    class_scores[class_idx] <= $signed({1'b0, pop_acc, 8'd0}) - thresh_l3[class_idx];

                    if (class_idx == 4'd9) begin
                        state <= ARGMAX_INIT;
                    end else begin
                        class_idx <= class_idx + 4'd1;
                        bit_idx   <= 10'd0;
                        pop_acc   <= 11'd0;
                        state     <= L3_FETCH;
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
    assign led = pred_digit;
    assign done = (state == DONE);

endmodule
