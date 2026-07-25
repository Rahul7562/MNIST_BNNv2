# ZedBoard defaults
set_property PACKAGE_PIN Y9 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

set_property PACKAGE_PIN T22 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN T21 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN U22 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN U21 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]

set_property PACKAGE_PIN V22 [get_ports done]
set_property IOSTANDARD LVCMOS33 [get_ports done]

set_property PACKAGE_PIN T18 [get_ports rst]
set_property IOSTANDARD LVCMOS18 [get_ports rst]

set_property PACKAGE_PIN R18 [get_ports start]
set_property IOSTANDARD LVCMOS18 [get_ports start]
