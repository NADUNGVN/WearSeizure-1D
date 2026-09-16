"""Activation memory under the supervisor's definition, plus what the RTL allocates.

    python scripts/memory_footprint.py [--bits 8]

The definition asked for: **the largest sum of the feature maps of two adjacent
layers**, plus any skip connection that must stay live across them. That is the
right question for a streaming accelerator, because a layer reads its input
while writing its output, so both must be resident at the same instant.

This network has **no skip connections**. `DepthwiseSeparableConv1d` is
depthwise -> pointwise -> BN -> ReLU and `MultiScaleDilatedBlock` in `k5_only`
mode is a single branch; neither adds its input back. So the sum of two adjacent
maps is the whole of it, with nothing to add.

Two numbers come out, and they answer different questions:

  peak adjacent pair   the minimum any implementation needs, and the number the
                       supervisor asked for
  ping-pong allocation what THIS accelerator actually reserves: two banks sized
                       to the largest map each ever holds, because layers
                       alternate between them rather than being packed

The second is always the larger. Quoting the first as the hardware requirement
would understate what is on the die; quoting the second as the architectural
requirement would overstate what the network needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "AI-Accelerator-RTL" / "model" / "manifest.json"

# CNN_1D_Core.v: FM_BANK_NUM = 16, FM_AWIDTH = 10 -> 16 x 1024 cells per buffer,
# and there are two of them (ping and pong).
FM_CELLS_PER_BUFFER = 16 * 1024


def layer_shapes(manifest: dict) -> list[tuple[str, int, int, int, int]]:
    """(name, in_ch, in_len, out_ch, out_len) for every layer, in order."""
    layers = list(manifest.get("layers", [])) + list(manifest.get("software_layers", []))
    layers.sort(key=lambda d: d["layer_id"])
    return [(d["name"], d["in_channels"], d["in_length"],
             d["out_channels"], d["out_length"]) for d in layers]


def footprint(shapes, bits: int, in_channels: int | None = None) -> dict:
    """Peak adjacent-pair activation, and the ping-pong allocation."""
    byte = bits / 8
    rows, peak, peak_at = [], 0.0, ""
    largest_map = 0.0
    for i, (name, ic, il, oc, ol) in enumerate(shapes):
        if i == 0 and in_channels is not None:
            ic = in_channels          # the ablation arms differ only at the stem
        fin, fout = ic * il * byte, oc * ol * byte
        rows.append((name, fin, fout, fin + fout))
        largest_map = max(largest_map, fin, fout)
        if fin + fout > peak:
            peak, peak_at = fin + fout, name
    return {
        "rows": rows,
        "peak_pair_bytes": peak,
        "peak_at": peak_at,
        "largest_map_bytes": largest_map,
        "ping_pong_bytes": 2 * largest_map,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shapes = layer_shapes(manifest)
    base = footprint(shapes, args.bits)

    print(f"DFP{args.bits}, {len(shapes)} layers, no skip connections\n")
    print(f"{'layer':<15}{'input':>10}{'output':>10}{'pair':>10}")
    for name, fin, fout, pair in base["rows"]:
        mark = "  <- peak" if name == base["peak_at"] else ""
        print(f"{name:<15}{fin:>10,.0f}{fout:>10,.0f}{pair:>10,.0f}{mark}")

    print(f"\npeak adjacent pair : {base['peak_pair_bytes']:,.0f} B "
          f"= {base['peak_pair_bytes'] / 1024:.2f} KiB   (at {base['peak_at']})")
    print(f"largest single map : {base['largest_map_bytes']:,.0f} B "
          f"= {base['largest_map_bytes'] / 1024:.2f} KiB")
    print(f"ping-pong allocated: {base['ping_pong_bytes']:,.0f} B "
          f"= {base['ping_pong_bytes'] / 1024:.2f} KiB")

    print(f"\nthe three channel-ablation arms, same network, stem input only:")
    print(f"{'arm':>6}{'peak pair':>14}{'ping-pong':>14}{'fits 16 Ki cells?':>20}")
    for ch in (1, 4, 18):
        f = footprint(shapes, args.bits, in_channels=ch)
        cells = ch * shapes[0][2]
        print(f"{ch:>4}ch{f['peak_pair_bytes'] / 1024:>12.2f} K"
              f"{f['ping_pong_bytes'] / 1024:>12.2f} K"
              f"{('yes' if cells <= FM_CELLS_PER_BUFFER else f'NO ({cells:,} cells)'):>20}")

    # Biases are stored at 32 bit, not at the data width: PE.v loads one as the
    # accumulator's initial value, so it lives at the accumulator's fixed point.
    # Counting them at 8 bit would understate the memory by about 2 KiB.
    all_layers = (list(manifest.get("layers", []))
                  + list(manifest.get("software_layers", [])))
    weights = manifest.get("total_weights", 0)
    biases = sum(d["out_channels"] for d in all_layers if d["type"] != "gap")
    w_kib = weights * args.bits / 8 / 1024
    b_kib = biases * 32 / 8 / 1024
    act_kib = base["peak_pair_bytes"] / 1024

    print(f"\n{'weights':<26}{weights:>8,} x {args.bits:>2} bit {w_kib:>8.2f} KiB")
    print(f"{'biases':<26}{biases:>8,} x 32 bit {b_kib:>8.2f} KiB")
    print(f"{'activations (peak pair)':<26}{'':>19}{act_kib:>8.2f} KiB")
    print(f"{'TOTAL on-chip':<26}{'':>19}{w_kib + b_kib + act_kib:>8.2f} KiB")
    print(f"\nwith the ping-pong allocation instead of the peak pair: "
          f"{w_kib + b_kib + base['ping_pong_bytes'] / 1024:.2f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
