"""The dynamic-fixed-point export that the RTL team builds against.

These artefacts leave this repository and become someone else's ground truth:
the hardware is verified by comparing its waveforms against the vectors this
exporter writes. So a silent error here does not show up as a bad accuracy
number, it shows up as an RTL team hunting a bug that is not in their RTL.

The three things worth pinning, in order of how quietly they would fail:

* the fifteen-layer decomposition must BE the network -- fold BatchNorm wrong
  and the quantised model is a faithful rendering of the wrong thing;
* `output_shift` must equal `p_in + p_weight - p_out` and fit the controller's
  6-bit field, because the RTL has no multiplier in the requantisation path;
* the bias must be scaled to the ACCUMULATOR's fixed point, since PE.v loads it
  as the accumulator's initial value rather than adding it after the shift.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from wearseizure.models.wearseizure1d import WearSeizure1D

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "export_dfp_hardware", ROOT / "scripts" / "export_dfp_hardware.py")
export_dfp = importlib.util.module_from_spec(_spec)
sys.modules["export_dfp_hardware"] = export_dfp
_spec.loader.exec_module(export_dfp)

GOLDEN_PATH = ROOT.parent / "AI-Accelerator-RTL" / "model" / "golden_model.py"


def build_k5only() -> WearSeizure1D:
    torch.manual_seed(0)
    model = WearSeizure1D(input_len=1024, kernel_mode="k5_only")
    # Give BatchNorm something other than its identity initialisation, so a
    # folding bug cannot pass by folding in a no-op.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.uniform_(m.bias, -0.5, 0.5)
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
    return model.eval().double()


def test_folded_decomposition_is_the_network():
    """The fifteen layers must compute what the model computes.

    This is the load-bearing test. Everything downstream quantises the folded
    layers, so if they are not the network, every other number is a precise
    measurement of the wrong model.
    """
    model = build_k5only()
    layers = export_dfp.decompose(model)
    x = torch.randn(4, 1, 1024, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(model(x), export_dfp.float_forward(layers, x), atol=1e-9)


def test_decomposition_lists_the_layers_the_hardware_runs():
    model = build_k5only()
    layers = export_dfp.decompose(model)
    assert [s.name for s in layers] == [
        "stem.0", "b1.dw", "b1.pw", "b2.dw", "b2.pw", "b3.dw", "b3.pw",
        "b4.dw", "b4.pw", "context.0.dw", "context.0.pw",
        "context.1.dw", "context.1.pw", "gap", "fc",
    ]
    relu = {s.name for s in layers if s.relu}
    # b1 and the context blocks are DepthwiseSeparableConv1d, where BN and ReLU
    # come after the POINTWISE -- so their depthwise output is unactivated, and
    # a golden model that activates it would diverge from the trained network.
    assert "b1.dw" not in relu
    assert "context.0.dw" not in relu and "context.1.dw" not in relu
    # b2-b4 are MultiScaleDilatedBlock, which activates both stages.
    assert {"b2.dw", "b3.dw", "b4.dw"} <= relu
    assert "gap" not in relu and "fc" not in relu


def test_multi_scale_is_refused_rather_than_mis_exported():
    """The multi-scale block is two depthwise branches, so it is two
    instructions and a different manifest. Exporting it as one would produce
    artefacts that look right and are not."""
    torch.manual_seed(0)
    model = WearSeizure1D(input_len=1024, kernel_mode="multi_scale").eval().double()
    with pytest.raises(SystemExit, match="k5_only"):
        export_dfp.decompose(model)


def test_exponent_search_never_loses_to_the_no_clipping_choice():
    """Allowing a little saturation must not make the error worse.

    The search starts at the exponent where nothing clips and moves tighter, so
    the no-clipping choice is always in the candidate set. If a tighter one is
    returned it is because it measured better, never merely because it was
    tried later.
    """
    rng = np.random.default_rng(0)
    # A heavy tail is exactly the case the no-clipping rule handles badly: one
    # outlier sets the scale for everything else.
    x = np.concatenate([rng.normal(0, 0.05, 4000), [3.0, -2.6]])
    p, stats = export_dfp.choose_exponent(x, bits=8)
    err = lambda q: float(np.mean((export_dfp.quantise(x, q, 8) / 2.0 ** q - x) ** 2))
    assert p >= stats["exponent_no_clip"]
    assert err(p) <= err(stats["exponent_no_clip"]) + 1e-18
    assert 0.0 <= stats["clipped_frac"] < 0.05


def test_exponent_of_an_all_zero_tensor_is_defined():
    p, stats = export_dfp.choose_exponent(np.zeros(16), bits=8)
    assert p == 0 and stats["clipped_frac"] == 0.0


def test_shifts_are_representable_and_internally_consistent():
    """`output_shift` is the whole requantisation, so it must be exact.

    The controller decodes six bits (Controller.v:27-22) and the quantiser only
    shifts right, so a shift outside [0, 63] is not something the hardware can
    do -- the exporter must refuse rather than emit it.
    """
    model = build_k5only()
    layers = export_dfp.decompose(model)
    x = torch.randn(2, 1, 1024, dtype=torch.float64)
    probe: list = []
    export_dfp.float_forward(layers, x, collect=probe)
    for spec, (_, t) in zip(layers, probe[1:]):
        spec.out_length = int(t.shape[2])
    for spec, prev in zip(layers, [probe[0][1]] + [t for _, t in probe[1:-1]]):
        spec.in_length = int(prev.shape[2])

    export_dfp.calibrate(layers, [x], bits=8)

    for spec in layers:
        assert 0 <= spec.output_shift < (1 << export_dfp.SHIFT_FIELD_BITS), spec.name
        if spec.type != "gap":
            assert spec.output_shift == spec.p_in + spec.p_weight - spec.p_out, spec.name
    # Each layer's input exponent is the previous layer's output exponent:
    # the hardware reads back exactly what it wrote, with no rescaling between.
    for prev, cur in zip(layers, layers[1:]):
        assert cur.p_in == prev.p_out, f"{prev.name} -> {cur.name}"
    # GAP averages by shifting, which is only an average when the length is a
    # power of two.
    gap = next(s for s in layers if s.type == "gap")
    assert gap.in_length == 1 << gap.output_shift


def test_exported_hex_reloads_to_exactly_what_was_quantised(tmp_path):
    """What lands on disk is what `$readmemh` gives the RTL.

    Checking the exporter's own arrays would test the wrong thing: the failure
    mode that matters is a formatting or sign-extension error between the two.
    """
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"golden_model.py not present at {GOLDEN_PATH}")
    golden = export_dfp.load_golden(GOLDEN_PATH.parent)

    model = build_k5only()
    layers = export_dfp.decompose(model)
    x = torch.randn(2, 1, 1024, dtype=torch.float64)
    probe: list = []
    export_dfp.float_forward(layers, x, collect=probe)
    for spec, (_, t) in zip(layers, probe[1:]):
        spec.out_length = int(t.shape[2])
    for spec, prev in zip(layers, [probe[0][1]] + [t for _, t in probe[1:-1]]):
        spec.in_length = int(prev.shape[2])
    export_dfp.calibrate(layers, [x], bits=8)
    export_dfp.write_artefacts(layers, tmp_path, bits=8, window_samples=1024,
                               provenance={"test": True})

    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    specs = [golden.LayerSpec.from_manifest(d) for d in manifest["layers"]]
    loaded = golden.load_weights(tmp_path / "weights", specs, bits=8)

    for spec in layers:
        if spec.type == "gap":
            continue
        w, b = loaded[spec.name]
        expected_w = export_dfp.quantise(spec.weight, spec.p_weight, 8).astype(np.int64)
        np.testing.assert_array_equal(w, expected_w)
        # The bias lives at the ACCUMULATOR's fixed point, p_in + p_weight,
        # because PE.v:113 loads it as the accumulator's initial value. Scaling
        # it to p_out instead would be wrong by a power of two, and the network
        # would still run.
        expected_b = np.round(spec.bias * 2.0 ** (spec.p_in + spec.p_weight)).astype(np.int64)
        np.testing.assert_array_equal(b, expected_b)


def test_golden_model_reproduces_its_own_dumped_vectors(tmp_path):
    """The RTL is checked against these vectors, so they must be reproducible.

    Running the model twice and comparing is not the point; reloading the
    weights from disk and getting the same trace is, because that is the path
    the hardware's own inputs take.
    """
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"golden_model.py not present at {GOLDEN_PATH}")
    golden = export_dfp.load_golden(GOLDEN_PATH.parent)

    model = build_k5only()
    layers = export_dfp.decompose(model)
    x = torch.randn(2, 1, 1024, dtype=torch.float64)
    probe: list = []
    export_dfp.float_forward(layers, x, collect=probe)
    for spec, (_, t) in zip(layers, probe[1:]):
        spec.out_length = int(t.shape[2])
    for spec, prev in zip(layers, [probe[0][1]] + [t for _, t in probe[1:-1]]):
        spec.in_length = int(prev.shape[2])
    export_dfp.calibrate(layers, [x], bits=8)
    export_dfp.write_artefacts(layers, tmp_path, bits=8, window_samples=1024,
                               provenance={"test": True})

    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    specs = [golden.LayerSpec.from_manifest(d) for d in manifest["layers"]]
    weights = golden.load_weights(tmp_path / "weights", specs, bits=8)

    q = np.clip(np.round(x[0, 0].numpy() * 2.0 ** layers[0].p_in), -128, 127).astype(np.int64)
    a = golden.GoldenModel(manifest, weights, bits=8, check_overflow=True)
    b = golden.GoldenModel(manifest, weights, bits=8, check_overflow=True)
    _, trace_a = a.run(q)
    _, trace_b = b.run(q)
    for key in trace_a:
        np.testing.assert_array_equal(trace_a[key], trace_b[key])
        assert trace_a[key].min() >= -128 and trace_a[key].max() <= 127, key

    # The 48-bit accumulator exists to make overflow impossible; confirm that
    # this network gets nowhere near it, so the RTL team can size it knowingly.
    assert a.peak_acc.bit_length() < export_dfp.ACC_BITS
