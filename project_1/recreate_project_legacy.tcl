# Recreate this design from sources without opening project_1.xpr.
#
# Usage:
#   vivado -mode batch -source recreate_project_legacy.tcl
#   vivado -mode batch -source recreate_project_legacy.tcl -tclargs <new_project_name> <part>
#
# Defaults:
#   new_project_name = project_1_legacy
#   part             = xc7z020clg484-1

set script_dir [file normalize [file dirname [info script]]]
set src_root   [file join $script_dir project_1.srcs]
set mem_dir    [file normalize [file join $script_dir mem_files]]

set project_name "project_1_legacy"
set part_name "xc7z020clg484-1"

if {[llength $argv] >= 1} {
    set project_name [lindex $argv 0]
}
if {[llength $argv] >= 2} {
    set part_name [lindex $argv 1]
}

set project_dir [file join $script_dir $project_name]

if {[file exists $project_dir]} {
    puts "ERROR: Project directory already exists: $project_dir"
    puts "Choose a new project name or delete the existing folder."
    exit 1
}

create_project $project_name $project_dir -part $part_name

# Design sources
set design_files [glob -nocomplain [file join $src_root sources_1 new *.sv]]
if {[llength $design_files] == 0} {
    puts "ERROR: No design sources found under [file join $src_root sources_1 new]"
    exit 1
}
add_files -fileset sources_1 $design_files
set_property top bnn_top [get_filesets sources_1]

# Constraints
set xdc_file [file join $src_root constrs_1 new zedboard_minimal.xdc]
if {![file exists $xdc_file]} {
    puts "ERROR: Constraint file not found: $xdc_file"
    exit 1
}
add_files -fileset constrs_1 $xdc_file

# Simulation testbench
set tb_file [file join $src_root sim_1 new bnn_tb.sv]
if {![file exists $tb_file]} {
    puts "ERROR: Testbench file not found: $tb_file"
    exit 1
}
add_files -fileset sim_1 $tb_file
set_property top bnn_tb [get_filesets sim_1]

# Memory initialization files used by both simulation and synthesis
set mem_files [glob -nocomplain [file join $mem_dir *.mem]]
if {[llength $mem_files] > 0} {
    add_files -fileset sim_1 $mem_files
    foreach mf $mem_files {
        set fobj [get_files -all $mf]
        if {[llength $fobj] > 0} {
            set_property USED_IN_SYNTHESIS true $fobj
            set_property USED_IN_SIMULATION true $fobj
        }
    }
}

# Match existing simulation setup used by bnn_tb.tcl
set mem_path_norm [string map {\\ /} $mem_dir]
set_property -name xsim.simulate.xsim.more_options -value "-d MEM_PATH=$mem_path_norm/" -objects [get_filesets sim_1]
set_property -name xsim.simulate.runtime -value "all" -objects [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

save_project
close_project

puts "DONE: Created project '$project_name' at $project_dir"
