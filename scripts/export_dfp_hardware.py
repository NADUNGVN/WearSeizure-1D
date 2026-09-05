"""Export the trained model as dynamic-fixed-point artefacts the RTL can load.

This is the model team's side of the handoff to `AI-Accelerator-RTL`. It
produces the four things that repository's `docs/MODEL_HANDOFF_GUIDE.md` asks
for: folded and quantised weights, a manifest with REAL per-layer shifts, one
set of per-layer test vectors, and the numbers needed to check them.

What makes this different from a generic quantiser
--------------------------------------------------
The accelerator is not a generic INT8 engine. It multiplies two integers,
accumulates in 48 bits, then requantises with a SINGLE ARITHMETIC SHIFT --
there is no multiplier in the requantisation path. So every scale must be a
power of two, and the shift is fully determined by three exponents:

    output_shift = p_in + p_weight - p_out

where a tensor stored with exponent `p` represents the real value `q / 2**p`.
Choosing those exponents IS the quantisation, and it is what this script does.

Two consequences that shaped the code:

* The RTL has one 6-bit `OUTPUT_SHIFT` field per layer, so `p_weight` must be
  per-tensor, not per-channel. Per-channel scales were what the modelling side
  preferred; the single-fold run reports what per-tensor costs in dB instead of
  assuming it is fine, so the RTL team can decide with a number in hand.
* The hardware runs the depthwise and pointwise convolutions as separate
  instructions, so it requantises BETWEEN them. The trained network has no
  activation there and never saw that rounding. That extra quantisation point
  is real and is modelled here rather than wished away.

Because the evaluation protocol is patient-specific LOSO, there is no single
"the model": each of the 66 folds trains its own weights. So the two jobs are
kept separate.

Configuration is Hydra's, so the flags below take a leading `+` -- they add
keys the base config does not define.

    # artefacts for RTL bring-up, from one fold (the first, unless named)
    python scripts/export_dfp_hardware.py profile=server data=chbmit \\
        model=wearseizure1d_k5only train.run_tag=L8 \\
        +export.fold=chb01__chb01_03

    # the quotable accuracy number, every fold through the integer datapath
    python scripts/export_dfp_hardware.py profile=server data=chbmit \\
        model=wearseizure1d_k5only train.run_tag=L8 +export.evaluate=true

Also available: `+export.bits=16` switches the whole toolchain to DFP16,
`+export.out_dir=...` writes elsewhere, and `+export.raw_margin=true` scores
from the final 48-bit accumulator rather than the two requantised logits.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.models.factory import build_model
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing

log = get_logger(__name__)
bootstrap_env(sys.argv)

ACC_BITS = 48
BIAS_BITS = 32
SHIFT_FIELD_BITS = 6          # Controller.v: OUTPUT_SHIFT = instruction[27:22]


# --------------------------------------------------------------------------
# Decomposing the network into the fifteen layers the hardware executes
# --------------------------------------------------------------------------

def eval_arm_dir(bits: int, raw_margin: bool) -> str:
    """Where a scoring arm's per-fold results live.

    The two arms -- the requantised logits and the raw 48-bit accumulator --
    answer different questions and each takes hours to produce, so they must not
    land in the same directory. The second run would overwrite the first, and
    the loss would be silent.
    """
    return f"dfp{bits}" + ("_acc" if raw_margin else "")


def write_lf(path: Path, text: str) -> None:
    """Write with Unix line endings, whatever platform the exporter runs on.

    These files are read by Verilog `$readmemh` and by the RTL team's tools.
    Python's default newline translation turns every "\\n" into "\\r\\n" when the
    exporter runs on Windows, and a stray carriage return in a hex file is the
    kind of defect that looks like a hardware bug for a day before anyone
    thinks to open the file in a hex editor.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


@dataclass
class HwLayer:
    """One instruction's worth of work, with BatchNorm already folded in."""
    layer_id: int
    name: str
    type: str                 # conv1d | depthwise | pointwise | gap | linear
    weight: np.ndarray        # (out, in_per_group, k); empty for gap
    bias: np.ndarray          # (out,); zeros where the layer has none
    stride: int
    dilation: int
    padding: int
    relu: bool
    in_channels: int
    out_channels: int
    in_length: int = 0
    out_length: int = 0
    # Filled by calibration.
    p_weight: int = 0
    p_in: int = 0
    p_out: int = 0
    output_shift: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def kernel_size(self) -> int:
        return int(self.weight.shape[2]) if self.weight.size else 1

    @property
    def groups(self) -> int:
        return self.in_channels if self.type == "depthwise" else 1


def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d | None) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into the convolution that precedes it.

    Folding is exact, not an approximation: BN at inference is an affine map
    per channel, and an affine map after a convolution is another convolution.
    Doing it here means the hardware never needs a BatchNorm unit.
    """
    w = conv.weight.detach().cpu().double().numpy()
    b = (conv.bias.detach().cpu().double().numpy() if conv.bias is not None
         else np.zeros(w.shape[0], dtype=np.float64))
    if bn is None:
        return w, b
    gamma = bn.weight.detach().cpu().double().numpy()
    beta = bn.bias.detach().cpu().double().numpy()
    mean = bn.running_mean.detach().cpu().double().numpy()
    var = bn.running_var.detach().cpu().double().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    return w * scale.reshape(-1, 1, 1), (b - mean) * scale + beta


def decompose(model: nn.Module) -> list[HwLayer]:
    """Read the fifteen hardware layers off the model itself.

    Strides, dilations and paddings are taken from the modules rather than
    written down here. A hand-copied table is a second source of truth that
    drifts; reading the modules cannot disagree with the network it describes.
    """
    def conv_layer(idx, name, conv, bn, typ, relu) -> HwLayer:
        w, b = fold_bn(conv, bn)
        return HwLayer(
            layer_id=idx, name=name, type=typ, weight=w, bias=b,
            stride=int(conv.stride[0]), dilation=int(conv.dilation[0]),
            padding=int(conv.padding[0]), relu=relu,
            in_channels=int(conv.in_channels), out_channels=int(conv.out_channels),
        )

    layers: list[HwLayer] = [
        conv_layer(1, "stem.0", model.stem[0], model.stem[1], "conv1d", True),
        # b1 is DepthwiseSeparableConv1d: BN sits after the POINTWISE, so the
        # depthwise output is neither normalised nor activated.
        conv_layer(2, "b1.dw", model.b1.depthwise, None, "depthwise", False),
        conv_layer(3, "b1.pw", model.b1.pointwise, model.b1.bn, "pointwise", True),
    ]
    for i, block in enumerate((model.b2, model.b3, model.b4), start=2):
        if block.branch_k3 is not None:
            raise SystemExit(
                "this exporter targets the k5_only variant; the multi-scale block "
                "concatenates two depthwise branches, which is two instructions and "
                "a different manifest. Re-run with model=wearseizure1d_k5only."
            )
        # MultiScaleDilatedBlock: BN and ReLU after BOTH stages.
        layers.append(conv_layer(2 * i, f"b{i}.dw", block.branch_k5, block.bn_dw,
                                 "depthwise", True))
        layers.append(conv_layer(2 * i + 1, f"b{i}.pw", block.pointwise, block.bn_pw,
                                 "pointwise", True))
    for j, ctx in enumerate(model.context):
        layers.append(conv_layer(10 + 2 * j, f"context.{j}.dw", ctx.depthwise, None,
                                 "depthwise", False))
        layers.append(conv_layer(11 + 2 * j, f"context.{j}.pw", ctx.pointwise, ctx.bn,
                                 "pointwise", True))

    layers.append(HwLayer(
        layer_id=14, name="gap", type="gap",
        weight=np.zeros((0, 0, 0)), bias=np.zeros(0),
        stride=1, dilation=1, padding=0, relu=False,
        in_channels=model.classifier.in_features,
        out_channels=model.classifier.in_features,
    ))
    fc_w = model.classifier.weight.detach().cpu().double().numpy()
    layers.append(HwLayer(
        layer_id=15, name="fc", type="linear",
        weight=fc_w[:, :, None],
        bias=model.classifier.bias.detach().cpu().double().numpy(),
        stride=1, dilation=1, padding=0, relu=False,
        in_channels=int(fc_w.shape[1]), out_channels=int(fc_w.shape[0]),
    ))
    return layers


def float_forward(layers: list[HwLayer], x: torch.Tensor,
                  collect: list | None = None) -> torch.Tensor:
    """Run the folded layers in sequence, in float, exactly as listed.

    This exists to be checked against `model(x)`. If the two disagree, the
    decomposition or the folding is wrong, and everything downstream would be
    quantising the wrong network -- a failure that is otherwise invisible
    because the quantised result would still look plausible.
    """
    a = x
    if collect is not None:
        collect.append(("input", a))
    for spec in layers:
        if spec.type == "gap":
            a = a.mean(dim=2, keepdim=True)
        else:
            w = torch.from_numpy(spec.weight).to(a.dtype)
            b = torch.from_numpy(spec.bias).to(a.dtype)
            a = F.conv1d(a, w, b, stride=spec.stride, padding=spec.padding,
                         dilation=spec.dilation, groups=spec.groups)
            if spec.relu:
                a = F.relu(a)
        if collect is not None:
            collect.append((spec.name, a))
    return a.flatten(1)


# --------------------------------------------------------------------------
# Choosing the exponents
# --------------------------------------------------------------------------

def quantise(x: np.ndarray, p: int, bits: int) -> np.ndarray:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.round(x * (2.0 ** p)), lo, hi)


def choose_exponent(x: np.ndarray, bits: int, search: int = 4) -> tuple[int, dict]:
    """Pick the power-of-two exponent that minimises squared error.

    The obvious rule -- take the exponent that makes the largest magnitude just
    fit, so nothing ever clips -- is not the best one. A single outlier then
    sets the scale for every other value, and at 8 bits that can throw away
    two or three bits of resolution on the values that actually matter. The
    hardware saturates rather than wrapping, so a little clipping is safe, and
    trading it for resolution is usually a net gain.

    So: start from the no-clipping exponent and search a few steps tighter,
    keeping whichever minimises the error against the float tensor. The search
    is upward only, because going the other way just wastes range.
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return 0, {"peak": 0.0, "clipped_frac": 0.0, "sqnr_db": float("inf")}

    p_noclip = int(np.floor(np.log2(((1 << (bits - 1)) - 1) / peak)))
    best_p, best_err, best_stats = p_noclip, None, {}
    for p in range(p_noclip, p_noclip + search + 1):
        q = quantise(x, p, bits)
        err = float(np.mean((q / (2.0 ** p) - x) ** 2))
        if best_err is None or err < best_err:
            signal = float(np.mean(x ** 2))
            best_p, best_err = p, err
            best_stats = {
                "peak": peak,
                "clipped_frac": float(np.mean(np.abs(np.round(x * 2.0 ** p))
                                              > (1 << (bits - 1)) - 1)),
                "sqnr_db": 10 * np.log10(signal / err) if err > 0 else float("inf"),
                "exponent_no_clip": p_noclip,
            }
    return best_p, best_stats


def per_channel_gain(x: np.ndarray, bits: int) -> float:
    """How much SQNR a per-channel weight scale would buy, in dB.

    Reported rather than used: the RTL has one OUTPUT_SHIFT per layer, so
    per-channel weight exponents would need a per-channel shift table. This
    number is what turns that into a decision instead of a preference.
    """
    if x.size == 0:
        return 0.0
    flat = x.reshape(x.shape[0], -1)
    p_t, _ = choose_exponent(x, bits)
    err_t = np.mean((quantise(x, p_t, bits) / 2.0 ** p_t - x) ** 2)
    err_c = 0.0
    for c in range(flat.shape[0]):
        p_c, _ = choose_exponent(flat[c], bits)
        err_c += np.sum((quantise(flat[c], p_c, bits) / 2.0 ** p_c - flat[c]) ** 2)
    err_c /= flat.size
    if err_c <= 0 or err_t <= 0:
        return 0.0
    return float(10 * np.log10(err_t / err_c))


def calibrate(layers: list[HwLayer], batches: list[torch.Tensor], bits: int,
              report_per_channel: bool = False) -> None:
    """Fill in p_weight, p_in, p_out and output_shift for every layer.

    Activations are calibrated on the fold's VALIDATION windows. Calibrating on
    test would be a leak, and documenting that practice elsewhere is part of
    what this project exists to do.
    """
    acts: dict[str, list[np.ndarray]] = {}
    for xb in batches:
        collected: list = []
        float_forward(layers, xb.double(), collect=collected)
        for name, tensor in collected:
            acts.setdefault(name, []).append(tensor.detach().cpu().numpy().ravel())
    pooled = {k: np.concatenate(v) for k, v in acts.items()}

    p_in, _ = choose_exponent(pooled["input"], bits)
    input_stats = {"p_in": p_in}
    log.info(f"input exponent p={p_in} (range +/-{np.abs(pooled['input']).max():.3f})")

    for spec in layers:
        spec.p_in = p_in
        if spec.type == "gap":
            # Sum then shift by log2(L): the RTL has no divider, and the sum
            # does not change the fixed point, so the shift IS the division.
            shift = int(np.log2(spec.in_length)) if spec.in_length else 0
            if spec.in_length != (1 << shift):
                raise SystemExit(
                    f"GAP input length {spec.in_length} is not a power of two, so the "
                    "average cannot be a shift. The hardware has no divider."
                )
            spec.p_weight, spec.p_out, spec.output_shift = 0, p_in, shift
        else:
            spec.p_weight, w_stats = choose_exponent(spec.weight, bits)
            spec.p_out, a_stats = choose_exponent(pooled[spec.name], bits)
            spec.output_shift = spec.p_in + spec.p_weight - spec.p_out
            spec.stats = {"weight": w_stats, "activation": a_stats}
            if report_per_channel:
                spec.stats["per_channel_gain_db"] = per_channel_gain(spec.weight, bits)

            if spec.output_shift < 0:
                raise SystemExit(
                    f"{spec.name}: output_shift is {spec.output_shift}, but the RTL only "
                    "shifts right. The output needs MORE fractional bits than the "
                    "accumulator carries, which means this layer's activations are tiny "
                    "relative to its inputs. Rescale the weights or widen the format."
                )
            if spec.output_shift >= (1 << SHIFT_FIELD_BITS):
                raise SystemExit(
                    f"{spec.name}: output_shift is {spec.output_shift}, which does not fit "
                    f"the controller's {SHIFT_FIELD_BITS}-bit OUTPUT_SHIFT field."
                )
        p_in = spec.p_out
    layers[0].stats.setdefault("input", input_stats)


# --------------------------------------------------------------------------
# Writing the artefacts
# --------------------------------------------------------------------------

def write_artefacts(layers: list[HwLayer], out_dir: Path, bits: int,
                    window_samples: int, provenance: dict) -> None:
    weights_dir = ensure_dir(out_dir / "weights")
    digits = bits // 4
    mask = (1 << bits) - 1
    total = 0

    for spec in layers:
        if spec.type == "gap":
            continue
        wq = quantise(spec.weight, spec.p_weight, bits).astype(np.int64)
        # The bias joins the accumulator directly (PE.v:113), so it is scaled
        # to the ACCUMULATOR's fixed point -- p_in + p_weight -- and not to the
        # output's. Getting this wrong is silent: the network still runs, the
        # biases are just in the wrong place by a power of two.
        bq = np.clip(np.round(spec.bias * 2.0 ** (spec.p_in + spec.p_weight)),
                     -(1 << (BIAS_BITS - 1)), (1 << (BIAS_BITS - 1)) - 1).astype(np.int64)

        safe = spec.name.replace(".", "_")
        write_lf(weights_dir / f"{spec.layer_id:02d}_{safe}_weights.txt",
                 "".join(f"{int(v) & mask:0{digits}X}\n" for v in wq.ravel()))
        write_lf(weights_dir / f"{spec.layer_id:02d}_{safe}_bias.txt",
                 "".join(f"{int(v) & 0xFFFFFFFF:08X}\n" for v in bq.ravel()))
        total += wq.size

    manifest = {
        "model_name": "WearSeizure-1D (k5_only)",
        "precision": f"DFP{bits}",
        "generated_by": "WearSeizure-1D/scripts/export_dfp_hardware.py",
        "provenance": provenance,
        "sampling_rate_hz": 256,
        "window_samples": window_samples,
        "num_classes": 2,
        "total_weights": total,
        "input_exponent": layers[0].p_in,
        "notes": [
            "A tensor stored with exponent p represents the real value q / 2**p.",
            "output_shift = p_in + p_weight - p_out, applied with round-to-nearest "
            "ties-away-from-zero, matching Fixed_Point_Quantizer.v.",
            "Biases are scaled to the ACCUMULATOR fixed point (p_in + p_weight), "
            "because PE.v loads the bias as the accumulator's initial value.",
            "`relu` is part of the network, not of the reference RTL, which has no "
            "activation unit yet.",
        ],
        "memory_config": {"num_banks": 16, "dwidth": bits, "bank_depth": 1024},
        "layers": [{
            "layer_id": s.layer_id, "name": s.name, "type": s.type,
            "in_channels": s.in_channels, "out_channels": s.out_channels,
            "in_length": s.in_length, "out_length": s.out_length,
            "kernel_size": s.kernel_size, "stride": s.stride,
            "dilation": s.dilation, "padding": s.padding,
            "relu": s.relu,
            "p_in": s.p_in, "p_weight": s.p_weight, "p_out": s.p_out,
            "output_shift": s.output_shift,
            "src_fm_sel": s.layer_id % 2 ^ 1, "dst_fm_sel": s.layer_id % 2,
            "calibration": s.stats,
        } for s in layers],
    }
    write_lf(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    # The golden model travels with the artefacts it describes. The RTL team
    # runs it without this repository checked out, and a golden model that is a
    # version behind the manifest is worse than none at all.
    if GOLDEN_SOURCE.is_file() and GOLDEN_SOURCE.resolve() != (out_dir / "golden_model.py").resolve():
        write_lf(out_dir / "golden_model.py", GOLDEN_SOURCE.read_text(encoding="utf-8"))

    log.info(f"wrote {total} weights, manifest.json and golden_model.py to {out_dir}")


# --------------------------------------------------------------------------
# The integer datapath, wrapped so the existing evaluator can score it
# --------------------------------------------------------------------------

class GoldenWrapper(nn.Module):
    """Runs the integer golden model but looks like the float model.

    Wrapping rather than reimplementing the metric: event matching, hysteresis,
    run-length and alarm timing are subtle and already written once. A second
    implementation here would measure a slightly different thing and nobody
    would notice which.
    """

    def __init__(self, golden, p_in: int, bits: int, raw_margin: bool,
                 logit_exponent: int, acc_exponent: int) -> None:
        super().__init__()
        self.golden, self.p_in, self.bits, self.raw_margin = golden, p_in, bits, raw_margin
        # The scale the returned numbers must be divided by to become the real
        # values the float model would have produced. This is not cosmetic:
        # engine_baseline scores with softmax(logits)[:, 1] and compares against
        # thresholds frozen from the FP32 run, so a logit that is 4x too large
        # sharpens the softmax and a raw accumulator 4096x too large saturates
        # it to exactly 0 or 1. Either way the frozen thresholds stop meaning
        # what they meant, and the result looks like a quantisation finding.
        self.scale = 2.0 ** (acc_exponent if raw_margin else logit_exponent)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().double().numpy()
        lo, hi = -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        q = np.clip(np.round(arr * 2.0 ** self.p_in), lo, hi).astype(np.int64)
        out = np.empty((q.shape[0], 2), dtype=np.float64)
        for i in range(q.shape[0]):
            logits, _ = self.golden.run(q[i, 0])
            if self.raw_margin:
                # The final accumulator, before it is squeezed into the data
                # width. Two int8 logits give the detector only ~8 bits of
                # score resolution, and the thresholds it compares against were
                # fitted on a continuous score -- so this is worth measuring
                # rather than assuming either way.
                acc = self.golden.final_acc.flatten()
                out[i] = [float(acc[0]), float(acc[1])]
            else:
                out[i] = [float(logits[0, 0]), float(logits[1, 0])]
        # Back to the real scale before the evaluator takes a softmax. What the
        # hardware stores is an integer; what it MEANS is that integer over
        # 2**exponent, and the meaning is what the thresholds were fitted on.
        return torch.from_numpy(out / self.scale).to(x.device, x.dtype)


GOLDEN_SOURCE = Path(__file__).resolve().parents[1] / "hardware" / "golden_model.py"


def report_score_agreement(golden, layers: list[HwLayer], model: nn.Module,
                           batches: list[torch.Tensor], bits: int,
                           limit: int = 256) -> None:
    """Does the integer datapath still rank windows the way the float model does?

    Detection does not use the logits directly: a threshold, hysteresis and a
    run-length filter turn a score sequence into events. What that pipeline
    needs is that the ORDER of the scores survives quantisation, and that the
    scores stay distinguishable enough for a threshold to sit between them.
    Neither is implied by low weight error, so both are measured here.

    The second row is the one to read. The RTL currently pushes the final layer
    through the same requantiser as every other, which squeezes the two logits
    into the data width and leaves the detector only a handful of distinct
    scores. Reading the 48-bit accumulator instead costs nothing in hardware --
    the value is already there -- and the two rows say what that is worth.
    """
    x = torch.cat(batches)[:limit]
    with torch.no_grad():
        fl = model(x).detach().cpu().numpy()
    float_margin = fl[:, 1] - fl[:, 0]

    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = np.clip(np.round(x[:, 0].numpy() * 2.0 ** layers[0].p_in), lo, hi).astype(np.int64)
    logit_margin, acc_margin = [], []
    for i in range(q.shape[0]):
        lg, _ = golden.run(q[i])
        logit_margin.append(float(lg[1, 0] - lg[0, 0]))
        a = golden.final_acc.flatten()
        acc_margin.append(float(a[1] - a[0]))

    def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\nscore agreement with the float model, {len(float_margin)} windows")
    spread = float(float_margin.max() - float_margin.min())
    scale = float(np.abs(float_margin).max()) + 1e-12
    print(f"  float margin spans [{float_margin.min():+.4f}, {float_margin.max():+.4f}]"
          f", spread {spread:.3e} ({100 * spread / scale:.2f}% of its magnitude)")
    # Relative, not absolute: a model whose scores all sit near -0.165 and vary
    # in the fifth decimal has no usable ranking either, and an absolute
    # threshold would wave it through.
    if spread / scale < 1e-3:
        # Without a spread in the reference there is no order to preserve, so a
        # rank correlation here would be a correlation between two orderings of
        # noise. Saying so beats printing a number that means nothing.
        print("  UNDETERMINED: the float model gives every window the same score, so "
              "there is\n  no ranking for quantisation to preserve. This says the "
              "checkpoint is degenerate,\n  not that the quantisation is good or bad. "
              "Re-run on a trained checkpoint.")
        return
    for label, g in (("int logits", np.array(logit_margin)),
                     ("48-bit accumulator", np.array(acc_margin))):
        print(f"  {label:<20} spearman {rank_corr(float_margin, g):+.4f}   "
              f"distinct scores {len(np.unique(g))}/{len(g)}   "
              f"same sign {100 * np.mean((float_margin > 0) == (g > 0)):.1f}%")
    print("  A low count of distinct scores means the detector cannot place a "
          "threshold\n  between windows the float model separates, however well "
          "the weights\n  were quantised. Re-run with +export.raw_margin=true to "
          "score from the\n  accumulator instead.")


def load_golden(_unused: Path | None = None):
    """Import the golden model from THIS repository, and deliver a copy.

    It lives here because this is where the network is defined and where it can
    be tested against the float model on every commit. The RTL repository gets
    a copy alongside the weights it describes -- a delivered artefact, not a
    second original. Keeping the source there instead meant the servers, which
    have this repository and the dataset but not that one, could not run it.
    """
    import importlib.util
    path = GOLDEN_SOURCE
    if not path.is_file():
        raise SystemExit(f"golden_model.py not found at {path}")
    spec = importlib.util.spec_from_file_location("golden_model", path)
    mod = importlib.util.module_from_spec(spec)
    # Registered before execution because @dataclass resolves annotations
    # through sys.modules, and a module loaded by path is not there yet.
    sys.modules["golden_model"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    bits = int(cfg.get("export", {}).get("bits", 8))
    out_dir = Path(cfg.get("export", {}).get(
        "out_dir", Path(__file__).resolve().parents[2] / "AI-Accelerator-RTL" / "model"))
    export_fold = cfg.get("export", {}).get("fold", None)
    do_evaluate = bool(cfg.get("export", {}).get("evaluate", False))
    raw_margin = bool(cfg.get("export", {}).get("raw_margin", False))
    n_cal_batches = int(cfg.get("export", {}).get("calibration_batches", 8))

    run_tag = run_tag_from_cfg(cfg)
    seed = seeds_from_cfg(cfg)[0]

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    records = load_records_from_manifest(
        manifest_df,
        data_dir=cfg.data.generated_dir if cfg.data.name == "synthetic" else None,
        raw_dir=cfg.data.raw_dir if cfg.data.name != "synthetic" else None,
    )
    folds = load_folds(str(Path(cfg.split.folds_path)),
                       expected_manifest_hash=hash_manifest(manifest_df))
    if cfg.train.get("max_folds"):
        folds = folds[: cfg.train.max_folds]
    art = Path(cfg.profile.artifacts_dir)
    src = fold_run_dir(art, cfg.model.name, cfg.split.name, cfg.window.name, seed, tag=run_tag)
    window_s, stride_s = cfg.window.window_s, cfg.window.stride_s

    if do_evaluate:
        # Fail now rather than after three hours of overwriting. A directory
        # already holding the other scoring arm's rows means another run is
        # writing here, and its results would be replaced silently.
        existing = art / "dfp_eval" / eval_arm_dir(bits, raw_margin) / f"seed{seed}"
        for path in sorted(existing.glob("*.json"))[:200]:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if bool(prior.get("raw_margin", False)) != raw_margin:
                raise SystemExit(
                    f"{path} was written by the other scoring arm "
                    f"(raw_margin={prior.get('raw_margin')}). Two runs are writing to "
                    "the same directory and one is destroying the other's results. "
                    "Stop the other run, then re-run the folds it overwrote."
                )

    chosen = [f for f in folds if f.fold_id == export_fold] if export_fold else folds[:1]
    if not chosen:
        raise SystemExit(f"fold {export_fold!r} is not in {cfg.split.folds_path}")

    golden_mod = load_golden(out_dir)
    results = []

    n_skipped = 0
    for fold in (folds if do_evaluate else chosen):
        if do_evaluate:
            # Resume rather than restart. A fold costs about three minutes and
            # there are 66 of them, so a run interrupted at fold 60 must not
            # begin again at fold 1. A resumed run does not rewrite the RTL
            # artefacts either -- those came from the first pass and are still
            # the same weights.
            done = (art / "dfp_eval" / eval_arm_dir(bits, raw_margin) / f"seed{seed}"
                    / f"{fold.fold_id}.json")
            if done.exists():
                n_skipped += 1
                continue

        ckpt = Path(src) / f"{fold.fold_id}.pt"
        metrics_path = Path(src) / f"{fold.fold_id}.metrics.json"
        if not ckpt.exists():
            raise SystemExit(
                f"missing {ckpt}. This script exports an ALREADY TRAINED model; it "
                "cannot invent weights."
            )
        model = build_model(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        model.eval().double()

        datasets, _band, _norm = build_fold_datasets(records, fold, window_s, stride_s)
        layers = decompose(model)

        with torch.no_grad():
            batches = []
            val = datasets["val"]
            step = max(1, len(val) // (n_cal_batches * 64))
            idx = list(range(0, len(val), step))[: n_cal_batches * 64]
            for start in range(0, len(idx), 64):
                chunk = [val[i][0] for i in idx[start:start + 64]]
                if chunk:
                    batches.append(torch.stack(chunk).double())
            if not batches:
                raise SystemExit(f"{fold.fold_id}: validation partition is empty")

            # Shapes come from a real forward pass, so the manifest can never
            # disagree with what the network actually produces.
            probe: list = []
            ref = model(batches[0])
            got = float_forward(layers, batches[0], collect=probe)
            for spec, (_, tensor) in zip(layers, probe[1:]):
                spec.out_length = int(tensor.shape[2])
            for spec, prev in zip(layers, [probe[0][1]] + [t for _, t in probe[1:-1]]):
                spec.in_length = int(prev.shape[2])

            drift = float((ref - got).abs().max())
            if drift > 1e-6:
                raise SystemExit(
                    f"{fold.fold_id}: the folded fifteen-layer decomposition disagrees "
                    f"with the model by {drift:.3e}. The BatchNorm folding or the layer "
                    "list is wrong, and quantising it would quantise the wrong network."
                )
            log.info(f"{fold.fold_id}: folded decomposition matches the model "
                     f"(max |delta| = {drift:.2e})")

            calibrate(layers, batches, bits, report_per_channel=not do_evaluate)

        if fold is chosen[0] or (export_fold and fold.fold_id == export_fold):
            write_artefacts(
                layers, out_dir, bits, int(batches[0].shape[2]),
                provenance={
                    "fold_id": fold.fold_id, "seed": seed, "run_tag": run_tag,
                    "model": cfg.model.name, "split": cfg.split.name,
                    "checkpoint": str(ckpt),
                    "calibrated_on": "validation partition of this fold",
                },
            )
            print("\nper-layer fixed point")
            print(f"{'layer':<14}{'p_in':>5}{'p_w':>5}{'p_out':>6}{'shift':>6}"
                  f"{'w SQNR':>9}{'a SQNR':>9}{'clip%':>7}")
            for s in layers:
                ws = s.stats.get("weight", {}).get("sqnr_db", float("nan"))
                a = s.stats.get("activation", {})
                print(f"{s.name:<14}{s.p_in:>5}{s.p_weight:>5}{s.p_out:>6}"
                      f"{s.output_shift:>6}{ws:>9.1f}"
                      f"{a.get('sqnr_db', float('nan')):>9.1f}"
                      f"{100 * a.get('clipped_frac', 0.0):>7.2f}")
            gains = [(s.name, s.stats.get("per_channel_gain_db", 0.0)) for s in layers
                     if "per_channel_gain_db" in s.stats]
            if gains:
                print("\nwhat per-channel weight scales would buy (dB SQNR), if the RTL")
                print("grew a per-channel shift table:")
                for name, g in sorted(gains, key=lambda t: -t[1])[:5]:
                    print(f"  {name:<14}{g:+.1f} dB")

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            weights = golden_mod.load_weights(
                out_dir / "weights",
                [golden_mod.LayerSpec.from_manifest(d) for d in manifest["layers"]],
                bits=bits)
            golden = golden_mod.GoldenModel(manifest, weights, bits=bits,
                                            check_overflow=True)
            x0 = batches[0][0, 0].numpy()
            q0 = np.clip(np.round(x0 * 2.0 ** layers[0].p_in),
                         -(1 << (bits - 1)), (1 << (bits - 1)) - 1).astype(np.int64)
            _, trace = golden.run(q0)
            vec_dir = ensure_dir(out_dir / "test_vectors")
            digits, mask = bits // 4, (1 << bits) - 1
            for i, (name, arr) in enumerate(trace.items()):
                fname = f"{i:02d}_{name.split('_', 1)[-1].replace('.', '_')}.txt"
                write_lf(vec_dir / fname,
                         "".join(f"{int(v) & mask:0{digits}X}\n" for v in arr.ravel()))
            print(f"\nwrote {len(trace)} test vectors to {vec_dir}")
            print(f"peak accumulator on this window: {golden.peak_acc} "
                  f"({golden.peak_acc.bit_length()} bits of {ACC_BITS})")
            report_score_agreement(golden, layers, model, batches, bits)

        if do_evaluate:
            params = json.loads(metrics_path.read_text(encoding="utf-8"))["frozen_postprocess"]["params"]
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            # Each fold has its OWN weights, so the manifest's shifts are
            # re-derived per fold; only the shapes are shared.
            for entry, spec in zip(manifest["layers"], layers):
                entry.update({"p_in": spec.p_in, "p_weight": spec.p_weight,
                              "p_out": spec.p_out, "output_shift": spec.output_shift,
                              "in_length": spec.in_length, "out_length": spec.out_length})
            tmp = ensure_dir(art / "dfp_export" / fold.fold_id)
            write_artefacts(layers, tmp, bits, int(batches[0].shape[2]),
                            provenance={"fold_id": fold.fold_id, "seed": seed})
            specs = [golden_mod.LayerSpec.from_manifest(d)
                     for d in json.loads((tmp / "manifest.json").read_text())["layers"]]
            golden = golden_mod.GoldenModel(
                json.loads((tmp / "manifest.json").read_text()),
                golden_mod.load_weights(tmp / "weights", specs, bits=bits), bits=bits)
            last = layers[-1]
            wrapper = GoldenWrapper(golden, layers[0].p_in, bits, raw_margin,
                                    logit_exponent=last.p_out,
                                    acc_exponent=last.p_in + last.p_weight)

            result = evaluate_fold(
                model=wrapper, records=records, fold=fold,
                window_s=window_s, stride_s=stride_s,
                postprocess_method=params["method"],
                postprocess_ema_alpha=params["ema_alpha"],
                postprocess_run_length=params["run_length"],
                postprocess_event_merge_gap_s=params["event_merge_gap_s"],
                threshold_on_grid=[params["threshold_on"]],
                threshold_off_grid=[params["threshold_off"]],
                batch_size=cfg.train.batch_size, device="cpu", num_workers=0,
                postprocess_alarm_timestamp=params.get("alarm_timestamp", "window_end"),
                datasets=datasets,
            )
            row = {"fold_id": fold.fold_id, "seed": seed, "bits": bits,
                   "raw_margin": raw_margin,
                   "sensitivity": result.test_event_metrics.sensitivity,
                   "far_per_hour": result.test_event_metrics.far_per_hour}
            results.append(row)
            out = ensure_dir(art / "dfp_eval" / eval_arm_dir(bits, raw_margin)
                             / f"seed{seed}")
            (out / f"{fold.fold_id}.json").write_text(json.dumps(row, indent=2),
                                                      encoding="utf-8")
            log.info(f"{fold.fold_id}: sens={row['sensitivity']:.4f} "
                     f"FAR={row['far_per_hour']:.4f}")

    if n_skipped:
        log.info(f"{n_skipped} folds already had results and were skipped; delete "
                 "their JSON to recompute them")

    if results:
        sens = float(np.mean([r["sensitivity"] for r in results]))
        far = float(np.mean([r["far_per_hour"] for r in results]))
        scope = "the folds computed in THIS run" if n_skipped else "all folds"
        print(f"\nDFP{bits} through the integer datapath, {len(results)} folds ({scope}):")
        print(f"  event sensitivity {sens:.4f}    FAR/h {far:.4f}")
        if n_skipped:
            print(f"  {n_skipped} further folds were already on disk and are NOT in "
                  "this mean.\n  Do not quote it -- it covers whatever this run happened "
                  "to recompute.")
        print("  For the cohort number, paired against FP32 and clustered by patient:")
        print(f"    python scripts/summarise_dfp_eval.py {art} --bits {bits}")


if __name__ == "__main__":
    main()
