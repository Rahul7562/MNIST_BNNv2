## Zedboard (xc7z020clg400-1) constraints for the MNIST BNN inference core.
## Clock: SYSCLK (Y9, 100 MHz, single-ended). Reset: btnC (N17, active-high).
## Drive all I/O with the slow LVCMOS33 standard.

# --- Clock (100 MHz) ---
set_property -dict {PACKAGE_PIN Y9  IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -add -name sys_clk -period 10.000 [get_ports clk]

# --- Reset (btnC, active-high) ---
set_property -dict {PACKAGE_PIN N17 IOSTANDARD LVCMOS33} [get_ports rst]
set_property -dict {PACKAGE_PIN N17 IOSTANDARD LVCMOS33} [get_ports {rst}]

# --- Control ---
set_property -dict {PACKAGE_PIN T18 IOSTANDARD LVCMOS33} [get_ports start]
set_property -dict {PACKAGE_PIN T17 IOSTANDARD LVCMOS33} [get_ports done]

# --- Output digit (4 bits) -> LEDs (LD0..LD3: T14, T15, P14, R14) ---
set_property -dict {PACKAGE_PIN T14 IOSTANDARD LVCMOS33} [get_ports {digit[0]}]
set_property -dict {PACKAGE_PIN T15 IOSTANDARD LVCMOS33} [get_ports {digit[1]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {digit[2]}]
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {digit[3]}]

# --- Input image (784 bits) : map a slice to GPIO for demo; full vector is
#     intended to be fed from a BRAM/DMA in a real system. For the standalone
#     core we expose image_in[783:0] and constrain only the lower bits to PMOD/JA
#     as a bring-up aid. Expand this list to wire all 784 bits to your board.
#     (Left as a documented template; the synthesis core itself has no I/O timing
#     dependency on how image_in is driven.)
# Example for the first 8 bits on PMOD JA (pins C17,C18,B17,B18,A17,A18,B14,B15):
# set_property -dict {PACKAGE_PIN C17 IOSTANDARD LVCMOS33} [get_ports {image_in[0]}]
# set_property -dict {PACKAGE_PIN C18 IOSTANDARD LVCMOS33} [get_ports {image_in[1]}]
# ... (repeat for all 784; typically driven from on-chip BRAM in deployment)

# --- Timing / implementation directives ---
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]

# Treat async reset synchronously at the boundary (deasserted cleanly in TB).
set_false_path -from [get_ports rst]

# The core is combinational-pipeline; allow a relaxed clock uncertainty.
set_clock_uncertainty 0.500 [get_clocks sys_clk]
