set project_name "bnn_mnist"
set part "xc7z020clg484-1"

if { $argc > 0 } { set part [lindex $argv 0] }

create_project ${project_name} ./vivado/${project_name} -part ${part} -force

add_files [glob -nocomplain ../hdl/*.sv]
add_files [glob -nocomplain ../mem_files/*.mem]

set xdc_file "./constraints.xdc"
if { [file exists $xdc_file] } {
    add_files -fileset constrs_1 $xdc_file
}

set_property top bnn_top [current_fileset]
update_compile_order -fileset sources_1

puts "Vivado project created successfully!"
