# RTL interface spec

Moved to keep the spec next to the code it constrains: see
[`../src/wearseizure/rtl_interface/spec.md`](../src/wearseizure/rtl_interface/spec.md)
for the datapath, bus, and software/RTL agreement points, and
[`../src/wearseizure/rtl_interface/golden_io_contract.py`](../src/wearseizure/rtl_interface/golden_io_contract.py)
for the machine-readable constants (window length, bit widths, register map
draft).

Status: spec only, no RTL yet. This work is deferred until the training
server and FPGA board (expected PYNQ-Z2 / Zynq-7020) are confirmed -- see the
README's "Server handoff checklist".
