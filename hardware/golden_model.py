#!/usr/bin/env python3
"""golden_model.py -- bit-exact fixed-point reference for WearSeizure-1D.

NumPy only. No PyTorch, no CUDA, no ML runtime: this is the numerical
specification the RTL testbench compares against, and it has to be runnable by
anyone debugging a waveform.

What it models, and why each detail matters
-------------------------------------------
Every arithmetic decision here is taken from the reference RTL rather than from
prose, because where the two disagree the RTL is what the hardware does:

  accumulator width 48 bits                CNN_1D_Core.v:23  ACC_DWIDTH = 48
  bias is the accumulator's INITIAL value  PE.v:113  accumulator_base_w =
                                           first_ifmap ? bias : accumulator_r
                                           -- so the bias is pre-scaled to
                                           p_in + p_weight, not to p_out
  requantise = ROUND then shift            Fixed_Point_Quantizer.v:22-58
  saturate to the data width               Fixed_Point_Quantizer.v:60-83

The rounding is the detail most likely to be got wrong. MODEL_HANDOFF_GUIDE.md
section 2.2 specifies a plain arithmetic right shift, which truncates. The RTL
does round-to-nearest with ties away from zero: it adds 2**(shift-1) before
shifting, or 2**(shift-1) - 1 when the accumulator is negative. Truncating
instead disagrees with the hardware on roughly half of all values by one LSB,
and fifteen layers of that does not stay small.

Arithmetic is int64 and therefore exact: the widest possible accumulation here
is 64 channels x 5 taps x 127 x 127 = 5.2e6, which is 23 bits. The 48-bit
hardware accumulator has enormous headroom, and --check-overflow proves that on
real data rather than assuming it.

Two things the reference RTL does NOT have
------------------------------------------
1. No activation function anywhere (grep -rni relu rtl/*.v finds nothing).
   This network needs ReLU after ten of its fifteen layers, listed in
   RELU_AFTER below. The golden model applies it; the RTL will have to grow it.
2. No dilation support. Four depthwise layers need dilation 2, 4, 8 and 16. The
   golden model implements dilation because the NETWORK has it; those layers
   cannot yet be checked against the RTL.

Both are recorded in model/MODEL_TEAM_TASKS.md rather than worked around here.

    python golden_model.py --self-test
    python golden_model.py --input test_vectors/00_input.txt --dump test_vectors/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent
MANIFEST_PATH = MODEL_DIR / "manifest.json"

# ReLU follows a layer when the trained network has one there. Read off the
# architecture, not guessed:
#   stem                      Conv - BN - ReLU
#   b1, context.0, context.1  DepthwiseSeparableConv1d: dw - pw - BN - ReLU,
#                             so the depthwise output is NOT activated
#   b2, b3, b4                MultiScaleDilatedBlock: dw - BN - ReLU - pw - BN
#                             - ReLU, so both are
#   gap, fc                   neither
RELU_AFTER = frozenset({
    "stem.0",
    "b1.pw",
    "b2.dw", "b2.pw",
    "b3.dw", "b3.pw",
    "b4.dw", "b4.pw",
    "context.0.pw",
    "context.1.pw",
})

ACC_BITS = 48


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    name: str
    type: str
    in_channels: int
    out_channels: int
    in_length: int
    out_length: int
    kernel_size: int
    stride: int
    dilation: int
    padding: int
    output_shift: int
    relu: bool

    @staticmethod
    def from_manifest(d: dict) -> "LayerSpec":
        return LayerSpec(
            layer_id=d["layer_id"], name=d["name"], type=d["type"],
            in_channels=d["in_channels"], out_channels=d["out_channels"],
            in_length=d["in_length"], out_length=d["out_length"],
            kernel_size=d["kernel_size"], stride=d["stride"],
            dilation=d["dilation"], padding=d["padding"],
            output_shift=d["output_shift"],
            # `relu` is written by the exporter. The fallback is the
            # architecture rather than False, so an older manifest cannot
            # silently produce a network with no activations at all.
            relu=bool(d.get("relu", d["name"] in RELU_AFTER)),
        )


def round_shift_signed(value: np.ndarray, shift: int) -> np.ndarray:
    """Round-to-nearest, ties away from zero, then arithmetic right shift.

    Transcribed from Fixed_Point_Quantizer.v. The asymmetric adjustment is not
    an approximation: for a negative accumulator the RTL adds 2**(shift-1) - 1
    rather than 2**(shift-1), which is what turns a floor-shift into a
    ties-away-from-zero round.
    """
    if shift == 0:
        return value
    if shift < 0:
        raise ValueError(f"negative shift {shift}: the RTL only shifts right")
    half = np.int64(1) << np.int64(shift - 1)
    adjust = np.where(value < 0, half - 1, half)
    return (value + adjust) >> np.int64(shift)


def saturate(value: np.ndarray, bits: int) -> np.ndarray:
    """Clamp to the two's-complement range of `bits`, as the RTL does.

    The range is [-2**(bits-1), 2**(bits-1)-1] -- asymmetric. The handoff guide
    calls this "symmetric saturation", which is the usual loose phrasing; the
    RTL's minimum really is -128 at 8 bits, not -127.
    """
    return np.clip(value, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def requantise(acc: np.ndarray, shift: int, bits: int, relu: bool) -> np.ndarray:
    """Accumulator -> stored activation: round-shift, saturate, then ReLU."""
    out = saturate(round_shift_signed(acc, shift), bits)
    return np.maximum(out, 0) if relu else out


def _patches(x: np.ndarray, spec: LayerSpec) -> np.ndarray:
    """(C, L) -> (C, out_length, K), gathering the dilated taps once.

    Zero padding is what the hardware sees while the line buffer has not
    filled, so padding with zeros models the hardware rather than merely being
    convenient.
    """
    if spec.padding:
        x = np.pad(x, ((0, 0), (spec.padding, spec.padding)))
    taps = np.arange(spec.kernel_size, dtype=np.int64) * spec.dilation
    starts = np.arange(spec.out_length, dtype=np.int64) * spec.stride
    cols = starts[:, None] + taps[None, :]
    if int(cols.max(initial=0)) >= x.shape[1]:
        raise ValueError(
            f"{spec.name}: needs input length {int(cols.max()) + 1} but has "
            f"{x.shape[1]} after padding {spec.padding}. The manifest's "
            "out_length, stride, dilation and padding are inconsistent."
        )
    return x[:, cols]


def conv_int(x: np.ndarray, w: np.ndarray, b: np.ndarray, spec: LayerSpec) -> np.ndarray:
    """Dense convolution: every output channel sees every input channel."""
    patch = _patches(x, spec)                                   # (Cin, T, K)
    acc = np.einsum("ctk,ock->ot", patch, w, optimize=True)     # (Cout, T)
    return acc + b[:, None]


def depthwise_int(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                  spec: LayerSpec) -> np.ndarray:
    """Depthwise convolution: one filter per channel, no mixing across channels.

    This is the layer type that makes per-channel scaling matter, because
    nothing here equalises one channel against another.
    """
    patch = _patches(x, spec)                                   # (C, T, K)
    acc = np.einsum("ctk,ck->ct", patch, w[:, 0, :], optimize=True)
    return acc + b[:, None]


def gap_int(x: np.ndarray) -> np.ndarray:
    """Global average pooling as a sum -- the division is the output shift.

    The RTL has no divider. Averaging over L samples is a sum followed by a
    right shift of log2(L), which is why the GAP layer's `output_shift` must be
    exactly that and not a calibrated value. At L = 32 the shift is 5.
    """
    return x.sum(axis=1, keepdims=True)


class GoldenModel:
    """The fifteen-layer datapath in integers, keeping every intermediate."""

    def __init__(self, manifest: dict, weights: dict, bits: int = 8,
                 check_overflow: bool = False) -> None:
        self.specs = [LayerSpec.from_manifest(d) for d in manifest["layers"]]
        self.weights = weights
        self.bits = bits
        self.check_overflow = check_overflow
        self.peak_acc = 0
        # The last layer's accumulator BEFORE it is squeezed into the data
        # width. Two int8 logits leave the detector only about eight bits of
        # score resolution, while the thresholds it compares against were
        # fitted on a continuous score, so whether that costs anything is worth
        # measuring. Keeping the accumulator here makes that measurable without
        # a second forward pass.
        self.final_acc = np.zeros((0, 0), dtype=np.int64)

    def run(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns (logits, {name: activation}); activations are as stored."""
        act = np.asarray(x, dtype=np.int64).reshape(1, -1)
        trace = {"00_input": act}

        for spec in self.specs:
            if spec.type == "gap":
                acc = gap_int(act)
            else:
                w, b = self.weights[spec.name]
                fn = depthwise_int if spec.type == "depthwise" else conv_int
                acc = fn(act, w, b, spec)

            self.peak_acc = max(self.peak_acc, int(np.abs(acc).max(initial=0)))
            if self.check_overflow and self.peak_acc >= (1 << (ACC_BITS - 1)):
                raise OverflowError(
                    f"{spec.name}: accumulator reached {self.peak_acc}, which does "
                    f"not fit the RTL's {ACC_BITS}-bit accumulator. The hardware "
                    "would wrap silently here."
                )

            if spec is self.specs[-1]:
                self.final_acc = acc

            act = requantise(acc, spec.output_shift, self.bits, spec.relu)
            trace[f"{spec.layer_id:02d}_{spec.name}"] = act

            if act.shape != (spec.out_channels, spec.out_length):
                raise ValueError(
                    f"{spec.name}: produced {act.shape}, manifest says "
                    f"({spec.out_channels}, {spec.out_length}). The manifest is the "
                    "contract the RTL is built from, so a mismatch here is real."
                )
        return act, trace

    def predict(self, x: np.ndarray) -> int:
        """Class-1 margin: logit[1] - logit[0], the score the thresholds see.

        A margin rather than a softmax on purpose -- softmax is monotone in the
        difference, so the difference carries the whole decision and the
        hardware never has to compute an exponential.
        """
        logits, _ = self.run(x)
        return int(logits[1, 0] - logits[0, 0])


def load_weights(weights_dir: Path, specs: list[LayerSpec], bits: int = 8) -> dict:
    """Read the exported hex files back as signed integers.

    Reading back rather than trusting the exporter's in-memory arrays is the
    point: it verifies that what landed on disk -- and so what $readmemh will
    load into the RTL -- is what the model intends.
    """
    def read_hex(path: Path, width: int) -> np.ndarray:
        vals = []
        for token in path.read_text(encoding="utf-8").split():
            v = int(token, 16)
            vals.append(v - (1 << width) if v >= (1 << (width - 1)) else v)
        return np.array(vals, dtype=np.int64)

    out = {}
    for spec in specs:
        if spec.type == "gap":
            continue
        safe = spec.name.replace(".", "_")
        wf = weights_dir / f"{spec.layer_id:02d}_{safe}_weights.txt"
        bf = weights_dir / f"{spec.layer_id:02d}_{safe}_bias.txt"
        if not wf.is_file():
            raise SystemExit(
                f"missing {wf}.\nRun the exporter in the WearSeizure-1D repo first:\n"
                "  python scripts/export_dfp_hardware.py ...\n"
                "This model reads exported hex; it does not invent weights."
            )
        cin = 1 if spec.type == "depthwise" else spec.in_channels
        w = read_hex(wf, bits).reshape(spec.out_channels, cin, spec.kernel_size)
        b = (read_hex(bf, 32) if bf.is_file()
             else np.zeros(spec.out_channels, dtype=np.int64))
        out[spec.name] = (w, b)
    return out


def self_test() -> int:
    """Check the arithmetic against cases worked by hand from the RTL.

    These are exactly the cases where a truncating implementation and the real
    one disagree, which is where a golden model goes wrong silently.
    """
    bad = []
    for value, shift, expected in [
        (3, 1, 2), (1, 1, 1), (2, 1, 1), (-1, 1, -1), (-3, 1, -2), (-2, 1, -1),
        (7, 2, 2), (-7, 2, -2), (6, 2, 2), (-6, 2, -2), (0, 5, 0), (100, 0, 100),
        (1 << 40, 20, 1 << 20), (-(1 << 40), 20, -(1 << 20)),
    ]:
        got = int(round_shift_signed(np.array([value], dtype=np.int64), shift)[0])
        if got != expected:
            bad.append(f"round_shift_signed({value}, {shift}) = {got}, want {expected}")

    if int(np.int64(3) >> np.int64(1)) == int(
            round_shift_signed(np.array([3], dtype=np.int64), 1)[0]):
        bad.append("rounding is indistinguishable from truncation -- the RTL rounds")

    for bits, value, expected in [(8, 200, 127), (8, -200, -128), (8, 127, 127),
                                  (8, -128, -128), (16, 40000, 32767),
                                  (16, -40000, -32768)]:
        got = int(saturate(np.array([value], dtype=np.int64), bits)[0])
        if got != expected:
            bad.append(f"saturate({value}, {bits}) = {got}, want {expected}")

    # ReLU must come from the architecture, not vanish with an absent field.
    spec = LayerSpec.from_manifest({
        "layer_id": 1, "name": "stem.0", "type": "conv1d", "in_channels": 1,
        "out_channels": 8, "in_length": 8, "out_length": 8, "kernel_size": 1,
        "stride": 1, "dilation": 1, "padding": 0, "output_shift": 0,
    })
    if not spec.relu:
        bad.append("stem.0 lost its ReLU when the manifest omitted the field")

    # A dilated layer must gather the taps the network actually asks for.
    dil = LayerSpec.from_manifest({
        "layer_id": 6, "name": "b3.dw", "type": "depthwise", "in_channels": 1,
        "out_channels": 1, "in_length": 9, "out_length": 1, "kernel_size": 3,
        "stride": 1, "dilation": 4, "padding": 0, "output_shift": 0, "relu": False,
    })
    got_taps = _patches(np.arange(9, dtype=np.int64).reshape(1, 9), dil).flatten().tolist()
    if got_taps != [0, 4, 8]:
        bad.append(f"dilation-4 gathered {got_taps}, want [0, 4, 8]")

    for line in bad:
        print("  FAIL:", line)
    print("self-test:", "FAILED" if bad else "all arithmetic checks pass")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=str(MODEL_DIR / "weights"))
    ap.add_argument("--input", help="hex file holding one input window")
    ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                    help="data width: 8 = DFP8, 16 = DFP16")
    ap.add_argument("--dump", help="write the 16 test vectors into this directory")
    ap.add_argument("--check-overflow", action="store_true",
                    help="fail if any accumulator would not fit 48 bits")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    specs = [LayerSpec.from_manifest(d) for d in manifest["layers"]]
    weights = load_weights(Path(args.weights), specs, bits=args.bits)

    if args.input:
        raw = [int(t, 16) for t in Path(args.input).read_text(encoding="utf-8").split()]
        half = 1 << (args.bits - 1)
        x = np.array([v - 2 * half if v >= half else v for v in raw], dtype=np.int64)
    else:
        print("[notice] no --input: running an impulse so the datapath is exercised.")
        print("         These are NOT the deliverable test vectors -- those need a")
        print("         real EEG window, exported alongside the weights.")
        x = np.zeros(manifest["window_samples"], dtype=np.int64)
        x[manifest["window_samples"] // 2] = 64

    model = GoldenModel(manifest, weights, bits=args.bits,
                        check_overflow=args.check_overflow)
    logits, trace = model.run(x)
    print(f"logits          : {logits.flatten().tolist()}")
    print(f"class-1 margin  : {int(logits[1, 0] - logits[0, 0])}")
    print(f"peak accumulator: {model.peak_acc} "
          f"({model.peak_acc.bit_length()} bits of {ACC_BITS} available)")

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        mask, digits = (1 << args.bits) - 1, args.bits // 4
        for i, (name, arr) in enumerate(trace.items()):
            fname = f"{i:02d}_{name.split('_', 1)[-1].replace('.', '_')}.txt"
            # Unix line endings whatever the platform: these are read by
            # Verilog $readmemh, and a stray carriage return in a hex file
            # looks like a hardware bug for a day.
            with open(out / fname, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(f"{int(v) & mask:0{digits}X}\n"
                                 for v in arr.flatten()))
        print(f"wrote {len(trace)} vectors to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
