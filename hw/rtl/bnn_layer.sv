`include "bnn_pkg.sv"

// One BNN layer, parameterized over input/output widths.
//
// Contract (ARCHITECTURE.md §5):
//   * Weights are stored as big-endian hex .mem files (bit 1 = +1). Read via $readmemh.
//   * For each output neuron j: P_j = popcount(XNOR(act_in, w_j)), z_j = 2*P_j - Nin.
//   * HIDDEN (IS_OUTPUT=0): act_out[j] = (P_j >= Th_j)  (integer threshold compare).
//   * OUTPUT (IS_OUTPUT=1): z_out[j]   = z_j  (integer dot); the argmax block turns
//                              this into a digit using per-class scale/offset.
//
// Computation is sequential (one neuron per clock) so the design is realistic
// (registered popcount + integer compare) and deterministic. `done` pulses for one
// cycle after the last neuron, then returns to idle.

module bnn_layer #(
    parameter int Nin  = 784,
    parameter int Nout = 256,
    parameter bit IS_OUTPUT = 1'b0
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [(Nin-1):0]       act_in,
    output logic [(Nout-1):0]      act_out,   // valid when IS_OUTPUT=0
    // Flattened z vector (Nout * 32 bits) — used when IS_OUTPUT=1.
    // (iverilog does not reliably drive packed-array output ports, so we flatten.)
    output logic [(Nout*32-1):0]   z_flat,    // valid when IS_OUTPUT=1
    output logic                   done
);

    `include "popcount.sv"

    // Weight BRAM (read-only after init).
    logic [(Nin-1):0] wmem [0:(Nout-1)];
    logic [15:0]      thmem [0:(Nout-1)];   // thresholds (only used when hidden)

    // Compile-time MEM_PATH is supplied via +define+MEM_PATH=...
    `ifndef MEM_PATH
        `define MEM_PATH "mem_files/"
    `endif

    generate
        if (IS_OUTPUT == 1'b0) begin : load_hidden
            initial begin
                $readmemh({`MEM_PATH, "layer", (Nin==784) ? "1" : "2", "_weights.mem"}, wmem);
                $readmemh({`MEM_PATH, "layer", (Nin==784) ? "1" : "2", "_thresholds.hex"}, thmem);
            end
        end else begin : load_output
            initial begin
                $readmemh({`MEM_PATH, "layer3_weights.mem"}, wmem);
            end
        end
    endgenerate

    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    logic [$clog2(Nout+1)-1:0] j;
    // Temporaries for the RUN computation (iverilog 12 rejects `automatic`
    // inside always blocks, so declare at module scope).
    logic [(Nin-1):0] x_run;
    int P_run;
    int z_run;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            j     <= 0;
            done  <= 1'b0;
            act_out <= '0;
            z_flat <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        j <= 0;
                        state <= RUN;
                    end
                end
                RUN: begin
                    x_run = ~(act_in ^ wmem[j]);
                    P_run = popcount(x_run, Nin);
                    z_run = 2 * P_run - Nin;
                    if (IS_OUTPUT == 1'b0) begin
                        // Store activation MSB-first (bit Nout-1 = neuron 0) to match the
                        // MSB-first convention of $readmemh weights and $fscanf inputs, so
                        // the next layer's popcount pairing stays consistent across layers.
                        act_out[Nout-1-j] <= (P_run >= thmem[j]) ? 1'b1 : 1'b0;
                    end else begin
                        z_flat[j*32 +: 32] <= z_run[31:0];
                    end
                    if (j == Nout-1) begin
                        state <= FINISH;
                    end else begin
                        j <= j + 1;
                    end
                end
                FINISH: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
