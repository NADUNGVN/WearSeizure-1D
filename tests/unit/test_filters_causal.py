from __future__ import annotations

import numpy as np

from wearseizure.signal.filters import CausalBandpass
from wearseizure.signal.normalize import AffineNormalizer, fit_affine_normalizer


def test_causal_bandpass_output_unaffected_by_future_samples():
    """Core causality property (memo 2.2): output at/before t must not depend
    on input after t. We change the signal only after t=5s and require the
    filtered output to be identical up to that point.
    """
    fs = 256.0
    n = int(10 * fs)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)

    cutover = int(5 * fs)
    x_variant = x.copy()
    x_variant[cutover:] = rng.standard_normal(n - cutover) * 100.0  # wildly different future

    f = CausalBandpass(fs_hz=fs)
    y = f.apply_full_reset(x)
    y_variant = f.apply_full_reset(x_variant)

    np.testing.assert_allclose(y[:cutover], y_variant[:cutover])


def test_causal_bandpass_state_chaining_matches_single_pass():
    """Filtering in two chunks with carried state must equal filtering the
    whole signal at once -- required so streaming (RTL) and offline (training)
    numerics can be reconciled.
    """
    fs = 256.0
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(4 * fs))
    split = int(1.5 * fs)

    f = CausalBandpass(fs_hz=fs)
    y_full = f.apply_full_reset(x)

    y1, state1 = f.apply(x[:split], state=None)
    y2, _ = f.apply(x[split:], state=state1)
    y_chunked = np.concatenate([y1, y2])

    np.testing.assert_allclose(y_full, y_chunked, atol=1e-10)


def test_causal_bandpass_rejects_invalid_band():
    import pytest

    with pytest.raises(ValueError):
        CausalBandpass(low_hz=30.0, high_hz=1.0, fs_hz=256.0)


def test_affine_normalizer_is_memoryless_and_matches_manual_formula():
    norm = AffineNormalizer(scale=2.0, bias=1.0)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(norm.apply(x), (x - 1.0) * 2.0)


def test_fit_affine_normalizer_uses_only_provided_train_signals():
    train_signal = np.array([0.0, 10.0, 20.0])
    norm = fit_affine_normalizer([train_signal], robust=False)
    assert norm.bias == np.mean(train_signal)
