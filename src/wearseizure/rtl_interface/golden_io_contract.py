"""Software-side declaration of the I/O contract the future RTL must match
(memo 4.5, 5.4). No RTL exists yet -- this module only pins down shapes,
dtypes, and register semantics in code so that:

1. `export_int8_reference.py` can serialize test vectors in exactly this
   format for the RTL testbench once it exists (Gate G3).
2. Changing this contract is a deliberate, reviewable diff, not something
   that drifts silently between the Python reference and the eventual RTL.

Nothing here executes on hardware; see `docs/RTL_INTERFACE_SPEC.md` for the
human-readable version of the same contract (AXI register map, timing).
"""
from __future__ import annotations

from dataclasses import dataclass

WINDOW_SAMPLES = 1024  # 4s @ 256 Hz (memo 4.1)
INPUT_SAMPLE_BITS = 16  # causal filter + affine normalize output (Table 4, Input stage)
WEIGHT_BITS = 8
ACTIVATION_BITS = 8
ACCUMULATOR_BITS = 32
NUM_CLASSES = 2

# AXI4-Lite control/telemetry register map (offsets in bytes from the IP's
# base address). Placeholder addresses -- to be finalized once RTL
# integration (Gate G3) fixes the actual register file.
class ControlRegister:
    CTRL = 0x00          # bit0: start, bit1: soft_reset
    STATUS = 0x04         # bit0: busy, bit1: alarm
    CYCLE_COUNT = 0x08     # telemetry: cycles since last reset
    INFERENCE_COUNT = 0x0C  # telemetry: windows processed
    ALARM_COUNT = 0x10      # telemetry: alarms raised
    OVERFLOW_COUNT = 0x14   # telemetry: accumulator saturation events
    WEIGHT_BASE_ADDR = 0x18  # where PS writes weights before start


@dataclass(frozen=True)
class StreamFrame:
    """One AXI-Stream beat carrying a single causally-filtered, normalized
    EEG sample (memo 4.5: AXI-Stream for sample input). `tlast` marks the
    final sample of a 1024-sample window.
    """

    sample: int  # INPUT_SAMPLE_BITS-wide signed value
    tlast: bool


@dataclass(frozen=True)
class InferenceResult:
    """What the PL is expected to make available for the PS to read back
    after a window completes (via AXI4-Lite or a result FIFO -- exact
    mechanism TBD once RTL integration starts).
    """

    logits: tuple[int, int]  # ACCUMULATOR_BITS-range, pre-softmax
    alarm: bool
    window_index: int
