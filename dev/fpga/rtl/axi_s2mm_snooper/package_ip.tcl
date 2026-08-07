# package_ip.tcl
#
# Run this from the Vivado TCL Console (not from the block design project —
# open a separate Vivado session or source it before opening the main project).
#
# Usage:
#   In Vivado TCL Console:
#       source /home/vivado/tmp/board-tools/axi_s2mm_snooper/package_ip.tcl
#
# After running, add the IP repo to the main project:
#   set_property ip_repo_paths [list /home/vivado/tmp/board-tools/axi_s2mm_snooper] [current_project]
#   update_ip_catalog

set script_dir [file dirname [info script]]
set ip_dir     $script_dir

# ---- Create a temporary project just for packaging ----
set tmp_proj /tmp/pkg_snooper_proj
file mkdir $tmp_proj

create_project -force pkg_snooper $tmp_proj -part xczu3eg-sbva484-1-i
add_files -norecurse [file join $ip_dir axi_s2mm_snooper.v]
update_compile_order -fileset sources_1
set_property top axi_s2mm_snooper [current_fileset]

# ---- Package as IP ----
ipx::package_project \
    -root_dir   $ip_dir \
    -vendor     user.org \
    -library    user \
    -taxonomy   /UserIP \
    -import_files \
    -force

set core [ipx::current_core]
set_property name            axi_s2mm_snooper          $core
set_property display_name   "AXI S2MM Snooper"         $core
set_property description    "Transparent AXI4 write passthrough with AXI-Lite debug registers. Insert between DMA M_AXI_S2MM and SmartConnect to capture AWLEN, WDATA per beat, WLAST position, and BRESP without affecting the data path." $core
set_property version         1.0                        $core
set_property core_revision   1                          $core
set_property supported_families { zynquplus Production } $core

# ---- Bus interface inference ----
# Vivado auto-detects S_AXI (write-only AXI4 slave), M_AXI (write-only AXI4
# master), and S_AXI_LITE (AXI4-Lite slave) from the port naming convention.
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core

# Associate aclk with all bus interfaces
foreach intf [ipx::get_bus_interfaces -of $core] {
    set ifname [get_property name $intf]
    ipx::add_bus_parameter ASSOCIATED_BUSIF \
        [ipx::get_bus_interfaces $ifname -of $core]
    # clock association
    set clk_assoc [ipx::get_bus_interfaces -filter \
        {VLNV =~ xilinx.com:signal:clock*} $core]
    if {[llength $clk_assoc] > 0} {
        set_property VALUE $ifname \
            [ipx::get_bus_parameters ASSOCIATED_BUSIF \
                -of [lindex $clk_assoc 0]]
    }
}

# Ensure aclk is flagged as the clock and aresetn as reset
ipx::infer_bus_interfaces xilinx.com:signal:clock_rtl:1.0   $core
ipx::infer_bus_interfaces xilinx.com:signal:reset_rtl:1.0   $core

# ---- Save and close ----
ipx::update_checksums $core
ipx::save_core $core

close_project -delete

puts ""
puts "==========================================="
puts " IP packaged at: $ip_dir"
puts " component.xml  written."
puts ""
puts " Next steps (in the main block design project):"
puts "   1. Settings -> IP -> Repository Manager -> add $ip_dir"
puts "   2. update_ip_catalog"
puts "   3. Add 'axi_s2mm_snooper' to block design"
puts "==========================================="
