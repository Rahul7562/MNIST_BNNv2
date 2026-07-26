VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal -sv --cc --exe --build

SIM = iverilog
SIM_FLAGS = -g2012

SRC_DIR = src
TB_DIR = tb
MEM_DIR = ../mem_files

# Define MEM_PATH for Verilog defines
DEFINE_MEM_PATH = -D MEM_PATH=\"$(MEM_DIR)/\"

srcs = $(wildcard $(SRC_DIR)/*.sv)
tb = $(TB_DIR)/bnn_tb.sv

compile:
	$(SIM) $(SIM_FLAGS) $(DEFINE_MEM_PATH) -I$(SRC_DIR) -o bnn_sim $(tb) $(srcs)

sim: compile
	vvp bnn_sim

clean:
	rm -f bnn_sim
	rm -rf obj_dir/
