set proj_dir [file normalize [file dirname [info script]]]
open_project [file join $proj_dir project_1.xpr]

set_property -name "xsim.simulate.runtime" \
  -value "all" \
  -objects [get_filesets sim_1]

launch_simulation

run all
