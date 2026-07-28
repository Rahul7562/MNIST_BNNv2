## Vivado project + synthesis + implementation flow for the MNIST BNN core.
##
## Usage (run on a machine with Vivado on PATH):
##   cd <repo> && vivado -mode batch -source hw/vivado/build.tcl -tclargs <mem_path>
## where <mem_path> is the directory containing the exported *.hex/*.mem files
## (default: hw/mem, produced by `bash hw/sim/run_sim.sh` / scripts/prepare_hw_mem.py).
##
## The RTL uses $readmemh to load weights/thresholds/scales/offsets from MEM_PATH.
## Vivado honors $readmemh for BRAM initialization, so no manual .coe is needed.
## Expected top: bnn_top. Target: Zedboard (xc7z020clg400-1).

set part "xc7z020clg400-1"

# MEM_PATH comes from the command line (default hw/mem).
if {[llength $argv] >= 1} {
    set mem_path [lindex $argv 0]
} else {
    set mem_path "hw/mem"
}
puts "INFO: MEM_PATH = $mem_path"

# Define the +define so the RTL's $readmemh points at the right directory.
set vlog_opts "+define+MEM_PATH=\"$mem_path/\""

create_project -force mnist_bnn_vivado ./vivado_project -part $part

# ---- RTL sources ----
set rtl_dir "hw/rtl"
add_files -norecurse [list \
    "$rtl_dir/bnn_pkg.sv" \
    "$rtl_dir/popcount.sv"  \
    "$rtl_dir/bnn_layer.sv" \
    "$rtl_dir/bnn_top.sv"]
set_property -name {include_dirs} -value {"$rtl_dir"} -objects [get_fileset sources_1]
# Pass the MEM_PATH define to the Verilog compilation.
set_property -name {VERILOG_DEFINE} -value "$vlog_opts" -objects [get_fileset sources_1]

# ---- Constraints ----
add_files -fileset constrs_1 -norecurse [list "hw/constraints/zedboard.xdc"]

# ---- Synthesis ----
set_property -name {STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY} -value "rebuilt" -objects [get_runs synth_1]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS} -value 1 -objects [get_runs synth_1]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1

# ---- Implementation ----
launch_runs impl_1 -jobs 4
wait_on_run impl_1
open_run impl_1

# ---- Timing report ----
report_timing_summary -file ./vivado_project/timing_summary.rpt
report_utilization -file ./vivado_project/utilization.rpt

# ---- Bitstream ----
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "INFO: DONE. Bitstream: ./vivado_project/mnist_bnn_vivado.runs/impl_1/bnn_top.bit"
puts "INFO: Utilization: ./vivado_project/utilization.rpt"
puts "INFO: Timing:      ./vivado_project/timing_summary.rpt"
