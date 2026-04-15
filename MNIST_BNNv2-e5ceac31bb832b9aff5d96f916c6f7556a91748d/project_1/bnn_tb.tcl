set proj_dir [file normalize [file dirname [info script]]]
open_project [file join $proj_dir project_1.xpr]

set_property -name "xsim.simulate.xsim.more_options" \
  -value "-d MEM_PATH=C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files/" \
  -objects [get_filesets sim_1]

launch_simulation

set xsim_dir [file normalize "[get_property DIRECTORY [current_project]]/project_1.sim/sim_1/behav/xsim"]
file copy -force "C:/Users/rahul/Desktop/Projects/MNIST_BNNv2/project_1/mem_files" $xsim_dir

run 1000ns
