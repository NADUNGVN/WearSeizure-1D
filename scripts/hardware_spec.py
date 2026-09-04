"""Per-layer specification for the accelerator team, derived from the model.

Everything here is read off the built network with forward hooks -- shapes,
kernels, dilations, MACs, and the streaming line buffer each layer needs. None
of it is transcribed by hand, because a spec handed to another team is exactly
where a hand-copied number becomes a fabricated one.

The line buffer is the figure that sizes on-chip SRAM. A 1D conv streaming one
sample at a time must hold the taps it will still need:

    (kernel_size - 1) * dilation + 1  input samples, x in_channels

so dilation, not kernel size, is what makes a layer expensive to buffer.

    python scripts/hardware_spec.py wearseizure1d_k5only [--markdown]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearseizure.models.factory import build_model


def spec_for(model_name: str, window_s: float = 4.0, fs_hz: int = 256):
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model" / f"{model_name}.yaml"
    cfg = OmegaConf.create({
        "model": OmegaConf.load(cfg_path),
        "window": {"window_s": window_s},
        "data": {"fs_hz": fs_hz},
    })
    model = build_model(cfg).eval()
    input_len = round(window_s * fs_hz)

    rows = []

    def hook(name):
        def fn(mod, inp, out):
            k = mod.kernel_size[0]
            d = mod.dilation[0]
            taps = (k - 1) * d + 1
            out_len = out.shape[-1]
            # One MAC per output element per input channel per tap, divided by
            # groups (a depthwise conv touches one input channel per output).
            macs = out.shape[1] * out_len * k * (mod.in_channels // mod.groups)
            rows.append({
                "layer": name,
                "type": "depthwise" if mod.groups == mod.in_channels != 1 else "conv",
                "in_ch": mod.in_channels, "out_ch": mod.out_channels,
                "k": k, "stride": mod.stride[0], "dilation": d,
                "in_len": inp[0].shape[-1], "out_len": out_len,
                "taps": taps,
                "buffer_elems": taps * mod.in_channels,
                "in_fmap": mod.in_channels * inp[0].shape[-1],
                "out_fmap": mod.out_channels * out_len,
                "macs": macs,
                "weights": sum(p.numel() for p in mod.parameters()),
            })
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model(torch.randn(1, cfg.model.in_channels, input_len))
    for h in handles:
        h.remove()

    fc = [m for m in model.modules() if isinstance(m, nn.Linear)]
    total_params = sum(p.numel() for p in model.parameters())
    return cfg, rows, fc, total_params, input_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="wearseizure1d_k5only")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    cfg, rows, fc, total_params, input_len = spec_for(args.model)
    total_macs = sum(r["macs"] for r in rows) + sum(m.in_features * m.out_features for m in fc)
    total_buf = sum(r["buffer_elems"] for r in rows)

    if args.markdown:
        print(f"### `{args.model}` -- per-layer specification\n")
        print(f"Input: {cfg.model.in_channels} channel x {input_len} samples "
              f"({cfg.window.window_s}s @ {cfg.data.fs_hz} Hz)\n")
        print("| layer | type | in | out | k | s | dil | in_len | out_len | taps | buffer | MACs | weights |")
        print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    else:
        print(f"{args.model}: input {cfg.model.in_channels}x{input_len}")
        print(f"{'layer':<28}{'type':<11}{'in':>4}{'out':>5}{'k':>3}{'s':>3}{'dil':>5}"
              f"{'out_len':>8}{'taps':>6}{'buffer':>8}{'MACs':>10}{'wts':>8}")
    for r in rows:
        if args.markdown:
            print(f"| `{r['layer']}` | {r['type']} | {r['in_ch']} | {r['out_ch']} | {r['k']} | "
                  f"{r['stride']} | {r['dilation']} | {r['in_len']} | {r['out_len']} | {r['taps']} | "
                  f"{r['buffer_elems']:,} | {r['macs']:,} | {r['weights']:,} |")
        else:
            print(f"{r['layer']:<28}{r['type']:<11}{r['in_ch']:>4}{r['out_ch']:>5}{r['k']:>3}"
                  f"{r['stride']:>3}{r['dilation']:>5}{r['out_len']:>8}{r['taps']:>6}"
                  f"{r['buffer_elems']:>8,}{r['macs']:>10,}{r['weights']:>8,}")

    biggest = max(rows, key=lambda r: r["buffer_elems"])

    # Two accelerator styles need two DIFFERENT memory numbers, and quoting only
    # one of them under-specifies the design.
    #
    #   fully streaming / pipelined -- every layer runs concurrently on the
    #     sample stream, so every line buffer is live at once and NO feature map
    #     is ever materialised. Memory = sum of line buffers.
    #
    #   layer sequential -- one layer at a time, so only that layer's line
    #     buffer is live, but the whole input AND output feature map must be
    #     stored. Memory = largest single line buffer + the largest adjacent
    #     feature-map pair.
    #
    # For this network the second is roughly twice the first, and neither is
    # obvious from the per-layer table alone.
    streaming = total_buf
    pair_peak = max(r["in_fmap"] + r["out_fmap"] for r in rows)
    # The TRUE peak is the largest (input + output + that layer's own buffer),
    # taken layer by layer. Adding the largest buffer to the largest pair
    # over-counts, because they occur at different layers: here the widest pair
    # is at b1.depthwise, whose buffer is 40 B, while the largest buffer belongs
    # to context.1.depthwise, whose feature maps are small.
    peak_row = max(rows, key=lambda r: r["in_fmap"] + r["out_fmap"] + r["buffer_elems"])
    sequential = peak_row["in_fmap"] + peak_row["out_fmap"] + peak_row["buffer_elems"]
    loose = biggest["buffer_elems"] + pair_peak
    print()
    print("activation memory, by accelerator style (INT8 bytes):")
    print(f"  {'fully streaming (all line buffers live, no feature maps)':<56}"
          f"{streaming:>8,} B  ({streaming/1024:.1f} KiB)")
    print(f"  {'layer sequential (peak of in + out + own buffer)':<56}"
          f"{sequential:>8,} B  ({sequential/1024:.1f} KiB)")
    print(f"  {'  peak occurs at':<56}{peak_row['layer']:>8}")
    print(f"  {'  widest adjacent feature-map pair, anywhere':<56}{pair_peak:>8,} B")
    print(f"  {'  largest single line buffer, anywhere':<56}"
          f"{biggest['buffer_elems']:>8,} B")
    print(f"  {'  (loose bound if those coincided -- they do not)':<56}{loose:>8,} B")
    # The same structure at every width. INT8 is the target, but the FP32
    # figure is what the model costs before any quantisation at all, and it is
    # the honest starting point for "how much does compression buy".
    print()
    print("footprint by numeric format (bytes; the network is identical, only the width changes):")
    print(f"  {'format':<14}{'weights':>12}{'streaming act':>16}{'seq act':>12}"
          f"{'TOTAL stream':>15}{'TOTAL seq':>13}")
    for label, width in (("FP32", 4), ("INT16 / DFP16", 2), ("INT8 / DFP8", 1)):
        w, sa, qa = total_params * width, streaming * width, sequential * width
        print(f"  {label:<14}{w:>10,} B{sa:>14,} B{qa:>10,} B"
              f"{w + sa:>13,} B{w + qa:>11,} B")
    print("  (as KiB)")
    for label, width in (("FP32", 4), ("INT16 / DFP16", 2), ("INT8 / DFP8", 1)):
        w, sa, qa = total_params * width, streaming * width, sequential * width
        print(f"  {label:<14}{w/1024:>10.1f}  {sa/1024:>13.1f}  {qa/1024:>9.1f}  "
              f"{(w+sa)/1024:>12.1f}  {(w+qa)/1024:>10.1f}")

    print()
    print("feature map after each layer (INT8 bytes):")
    for r in rows:
        print(f"  {r['layer']:<24}{r['out_ch']:>4} x {r['out_len']:<6}{r['out_fmap']:>8,} B")
    print()
    try:
        from thop import profile
        cfg2, _, _, _, _ = spec_for(args.model)
        thop_macs = int(profile(build_model(cfg2), inputs=(torch.randn(1, cfg.model.in_channels, input_len),),
                                verbose=False)[0])
    except ImportError:
        thop_macs = None

    print(f"{'conv+fc MACs (deployable)':<34}{total_macs:>12,}")
    if thop_macs:
        print(f"{'thop MACs (paper/gate figure)':<34}{thop_macs:>12,}")
        print(f"  The two differ by {thop_macs - total_macs:,}: thop counts BatchNorm and elementwise")
        print("  ops as well. BatchNorm FOLDS INTO the preceding convolution at inference, so the")
        print("  number the accelerator must actually issue is the conv+fc one. The gate and every")
        print("  cross-model comparison in the paper use the thop figure, consistently for all models.")
    print(f"{'total parameters':<34}{total_params:>12,}")
    print(f"{'INT8 weight memory':<34}{total_params:>12,} B  ({total_params/1024:.1f} KiB)")
    print(f"{'line-buffer elements (INT8 bytes)':<34}{total_buf:>12,} B  ({total_buf/1024:.1f} KiB)")
    print(f"largest single buffer: {biggest['layer']} = {biggest['buffer_elems']:,} elems "
          f"({100*biggest['buffer_elems']/total_buf:.0f}% of the total), "
          f"because k={biggest['k']} at dilation {biggest['dilation']} spans {biggest['taps']} samples "
          f"of a {biggest['in_len']}-sample sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
