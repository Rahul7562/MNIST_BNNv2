`ifndef MEM_PATH
`define MEM_PATH "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/"
`endif

module bnn_top (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [783:0] image_in,
    output logic [3:0]   pred_digit,
    output logic         done
);

    typedef enum logic [2:0] {
        IDLE,
        LAYER1,
        LAYER2,
        OUTPUT,
        DONE
    } state_t;

    state_t state;

    logic [783:0] image_reg;
    logic [511:0] l1_reg;
    logic [255:0] l2_reg;
    logic [3:0] pred_reg;

    logic [511:0] l1_comb;
    logic [255:0] l2_comb;
    logic [3:0] pred_comb;

    bnn_layer1 u_bnn_layer1 (
        .image_in(image_reg),
        .l1_out(l1_comb)
    );

    bnn_layer2 u_bnn_layer2 (
        .l1_in(l1_reg),
        .l2_out(l2_comb)
    );

    bnn_output u_bnn_output (
        .l2_in(l2_reg),
        .pred_digit(pred_comb)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            image_reg <= '0;
            l1_reg    <= '0;
            l2_reg    <= '0;
            pred_reg  <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        image_reg <= image_in;
                        state <= LAYER1;
                    end
                end

                LAYER1: begin
                    l1_reg <= l1_comb;
                    state <= LAYER2;
                end

                LAYER2: begin
                    l2_reg <= l2_comb;
                    state <= OUTPUT;
                end

                OUTPUT: begin
                    pred_reg <= pred_comb;
                    state <= DONE;
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
    assign done = (state == DONE);

endmodule
