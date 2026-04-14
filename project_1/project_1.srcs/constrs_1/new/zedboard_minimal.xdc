# FIX: Minimal FPGA constraints for bnn_top on ZedBoard (xc7z020clg484-1)

# FIX: 100 MHz onboard clock
set_property PACKAGE_PIN Y9 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
# FIX: timing target relaxed per requirement.
create_clock -period 20.0 [get_ports clk]

# FIX: Reset button (BTNU on Bank 34)
set_property PACKAGE_PIN T18 [get_ports rst]
set_property IOSTANDARD LVCMOS18 [get_ports rst]

# FIX: Start button (BTNR on Bank 34)
set_property PACKAGE_PIN R18 [get_ports start]
set_property IOSTANDARD LVCMOS18 [get_ports start]

# FIX: 4-bit predicted digit LEDs
set_property PACKAGE_PIN T22 [get_ports {led[0]}]
set_property PACKAGE_PIN T21 [get_ports {led[1]}]
set_property PACKAGE_PIN U22 [get_ports {led[2]}]
set_property PACKAGE_PIN U21 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

# FIX: Done LED
set_property PACKAGE_PIN V22 [get_ports done]
set_property IOSTANDARD LVCMOS33 [get_ports done]

# FIX: Optional external input timing constraints.
set_input_delay 2 -clock clk [get_ports rst]
set_input_delay 2 -clock clk [get_ports start]
