"""Reconstruct published architectures and compute their memory footprint.

Most papers report a parameter count and nothing about memory. That leaves the
comparison table with "NR" in the column that matters most for an accelerator,
which is not good enough when the architecture is described well enough to
rebuild.

Every reconstruction here is CHECKED against the paper's own reported parameter
count before its footprint is used. If the rebuild does not reproduce that
number, the reconstruction is wrong and the footprint is not reported -- the
check is the whole reason this is trustworthy rather than a guess.

    python scripts/estimate_published_footprint.py

Numbers are for INT8 unless stated; multiply by 2 for INT16, by 4 for FP32.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    in_ch: int
    out_ch: int
    k: int
    stride: int = 1
    pool: int = 1
    in_len: int = 0
    out_len: int = 0
    stage: int = 0                      # layers sharing a stage run CONCURRENTLY

    @property
    def weights(self) -> int:
        return self.in_ch * self.out_ch * self.k

    @property
    def bn(self) -> int:
        return 2 * self.out_ch          # scale and bias, folded at inference

    @property
    def in_fmap(self) -> int:
        return self.in_ch * self.in_len

    @property
    def out_fmap(self) -> int:
        return self.out_ch * self.out_len


@dataclass
class Reconstruction:
    name: str
    source: str
    input_ch: int
    input_len: int
    reported_params: int
    tolerance: float                    # fraction; papers round and omit biases
    layers: list[Layer] = field(default_factory=list)
    fc: list[tuple[int, int]] = field(default_factory=list)
    parallel_branches: int = 1
    notes: str = ""

    def params(self) -> int:
        conv = sum(x.weights + x.bn for x in self.layers)
        dense = sum(i * o + o for i, o in self.fc)
        return conv + dense

    def peak_activation(self) -> tuple[int, str]:
        """Largest live activation set, in values, taken STAGE by stage.

        Layers sharing a stage run concurrently on separate branches, so every
        one of their inputs and outputs is live at the same moment. Taking the
        maximum over individual layers instead would halve the answer for a
        two-branch network -- which is exactly the mistake this replaced.
        """
        best, where = 0, ""
        for stage in sorted({x.stage for x in self.layers}):
            group = [x for x in self.layers if x.stage == stage]
            # Branches at the same depth read the same tensor once, so a shared
            # input is counted once; distinct branch inputs are counted each.
            ins = {(x.in_ch, x.in_len, x.name.split(".")[-1] if stage > 1 else "shared") for x in group}
            live = sum(c * n for c, n, _ in ins) + sum(x.out_fmap for x in group)
            if live > best:
                best, where = live, f"stage {stage} ({len(group)} concurrent branches)"
        return best, where


def chung2024() -> Reconstruction:
    """Chung et al. 2024, Front. Neurol. 15:1389731.

    From the paper: 4-s windows at 256 Hz (1024 samples), one channel. Two
    PARALLEL modules, kernel 1x3 and 1x5, each with three conv/BN/max-pool
    layers of 32, 64 and 128 filters, strides 2, 2, 1 and pooling size 3.
    Concatenated, then global average pooling and fully connected layers.

    The FC stage is described only as "fully connected layers". A single
    256->2 layer gives about 83.6 k parameters, well short of the reported
    116.7 k; 256->128->2 gives 116.2 k, which matches. That is the
    reconstruction used, and the parameter check is what justifies it.
    """
    lens = []
    n = 1024
    for stride in (2, 2, 1):
        n = n // stride
        n = n // 3                      # pooling size 3
        lens.append(n)

    layers = []
    prev, prev_len = 1, 1024
    for i, (out, stride, out_len) in enumerate(zip((32, 64, 128), (2, 2, 1), lens), start=1):
        for k, tag in ((3, "k3"), (5, "k5")):
            layers.append(Layer(f"conv{i}.{tag}", prev, out, k, stride, 3, prev_len, out_len, stage=i))
        prev, prev_len = out, out_len

    return Reconstruction(
        name="Chung et al. 2024",
        source="Front. Neurol. 15:1389731, architecture from the paper text",
        input_ch=1, input_len=1024,
        reported_params=116_700, tolerance=0.02,
        layers=layers,
        fc=[(256, 128), (128, 2)],      # 256 = concat of the two 128-filter branches
        parallel_branches=1,            # both branches already listed explicitly
        notes="two parallel modules (1x3 and 1x5), concatenated, then GAP and FC",
    )


def report(rec: Reconstruction) -> None:
    got = rec.params()
    off = abs(got - rec.reported_params) / rec.reported_params
    ok = off <= rec.tolerance
    print(f"\n=== {rec.name} ===")
    print(f"  {rec.source}")
    print(f"  {rec.notes}")
    print(f"  input {rec.input_ch} x {rec.input_len}")
    print(f"  parameters: rebuilt {got:,} vs reported {rec.reported_params:,} "
          f"({off:+.1%})  -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("  reconstruction rejected; footprint NOT reported for this work")
        return

    peak, where = rec.peak_activation()
    print(f"  peak activation: {peak:,} values, at {where}")
    print(f"  {'format':<8}{'weights':>12}{'activations':>14}{'TOTAL':>12}{'% XC7Z020 BRAM':>18}")
    for label, width in (("INT8", 1), ("INT16", 2), ("FP32", 4)):
        w, a = got * width, peak * width
        pct = 100 * (w + a) / (140 * 4096)
        print(f"  {label:<8}{w/1024:>10.1f} KB{a/1024:>12.1f} KB{(w+a)/1024:>10.1f} KB{pct:>16.0f} %")


def main() -> int:
    for rec in (chung2024(),):
        report(rec)
    print("\nOther works in the comparison table still need their architectures read")
    print("out of the paper before a footprint can be computed rather than guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
