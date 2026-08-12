# RTL interface spec (draft, pre-hardware)

Status: **spec only** -- no RTL exists yet. This is the interface contract
the software pipeline is designed around so that when RTL work starts (Gate
G3, after the training server and FPGA board are confirmed), the datapath
described in memo section 4.5 has an unambiguous target to implement against.
See `golden_io_contract.py` for the machine-readable version of the same
constants.

## Datapath (memo 4.5)

1. **Tap generator**: shared across kernel sizes k=3/5 and dilations 1/2/4,
   time-multiplexed on one PE bank (no duplicated buffers per branch).
2. **Depthwise engine + pointwise MAC array**: `P_PE` parallelism
   configurable at 4/8/16 to trade resources for latency.
3. **Fused Conv-BN-ReLU-requant**: BatchNorm folded into the conv, requant
   scale is power-of-two where accuracy allows (shift instead of multiplier).
   Accumulator is INT32 with checked saturation.
4. **On-chip memory only**: ping-pong activation SRAM, ~16 KB of INT8 weights
   resident on-chip. No DDR access during inference after weights are loaded.
5. **Postprocessor**: EMA (alpha=1/8) + two-threshold hysteresis + run-length
   + event merge, entirely on-chip; thresholds are loaded, never computed, on
   the PL side (they are frozen on validation in software -- see
   `training/threshold_selection.py`).

## Buses

- **AXI4-Lite**: control/telemetry. Register map draft in
  `golden_io_contract.ControlRegister` -- start/reset, busy/alarm status,
  cycle/inference/alarm/overflow counters, weight base address.
- **AXI-Stream**: sample input, one `StreamFrame` per causally-filtered,
  normalized EEG sample, `tlast` on the 1024th sample of each 4s window.

## Software/RTL agreement points

- Window length: **1024 samples** (4s @ 256 Hz), matching
  `configs/window/w4s_stride1s.yaml`.
- Input sample precision: **16-bit**, output of the causal band-pass +
  affine normalize stage (Table 4 Input row) -- computed identically in
  `signal/filters.py` + `signal/normalize.py` and (eventually) in RTL.
- Weights/activations: **INT8**, accumulator **INT32** with saturation
  (memo Table 1), matching `quant/qat.py` and `quant/int_reference.py`.
- Bit-exact verification target (memo 5.4): 100% logit/class equality between
  `quant/int_reference.py` and RTL simulation on >=10,000 windows, plus
  dedicated tests for saturation, max/min, reset, packet gap, and
  backpressure -- none of which exist yet since there is no RTL to test
  against.

## Explicitly deferred until server/FPGA info is confirmed

- Exact register addresses and AXI-Lite timing.
- `P_PE` parallelism choice (depends on the actual board's DSP/BRAM budget --
  PYNQ-Z2/Zynq-7020 assumed per memo Table 1, 220 DSP / 630 KB BRAM, but not
  yet confirmed).
- Weight-loading protocol (DMA vs. AXI-Lite register writes).
- Clock domain / Fmax target validation (memo Table 6: >=100 MHz timing-closed).
