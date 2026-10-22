"""Lever L9: onset-proximity sampling must boost exactly the late windows.

The claim this lever rests on is narrow and measurable: the detector's errors
are false negatives in the first seconds after onset, so those windows should
carry more sampling probability. Two things have to hold for that to be worth
running, and neither is obvious from reading the code.

First, **off must be off**. Every result in the experiment log was produced
with the old sampler, and "equivalent weights" is not good enough -- a
different float path would make old folds irreproducible for no gain.

Second, the boost must land on windows chosen by their distance from an
*onset*, not by their label. A label says a window touches a seizure, which
cannot tell the first second of one from the ninetieth, and boosting the whole
ictal class is a different lever that does nothing for latency.
"""
from __future__ import annotations

import numpy as np

from wearseizure.data.manifest import SeizureEvent
from wearseizure.data.sampler import make_class_balanced_sampler, onset_proximity_boost


class _Record:
    def __init__(self, events):
        self.meta = type("M", (), {"seizure_events": events})()


class _Window:
    def __init__(self, edf_id, end_sec, label):
        self.edf_id, self.end_sec, self.label = edf_id, end_sec, label


class _Dataset:
    """Only what the sampler reads: `windows` and `records`."""

    def __init__(self, windows, records):
        self.windows, self.records = windows, records


def _toy():
    # One seizure, 100 s to 160 s. Windows every second from 95 s to 115 s.
    events = [SeizureEvent(event_id="e0", onset_sec=100.0, offset_sec=160.0)]
    records = {"r": _Record(events)}
    windows = [_Window("r", t, int(t > 100.0)) for t in np.arange(95.0, 116.0, 1.0)]
    return _Dataset(windows, records)


def test_boost_is_identically_one_when_off():
    d = _toy()
    for within, factor in ((0.0, 3.0), (10.0, 1.0), (0.0, 1.0)):
        b = onset_proximity_boost(d, within, factor)
        assert b.shape == (len(d.windows),)
        assert np.array_equal(b, np.ones(len(d.windows)))


def test_off_reproduces_the_old_sampler_weights_bit_for_bit():
    d = _toy()
    base = make_class_balanced_sampler(d)
    same = make_class_balanced_sampler(d, onset_within_s=0.0, onset_factor=1.0)
    a = np.asarray(base.weights, dtype=np.float64)
    b = np.asarray(same.weights, dtype=np.float64)
    # Bit-for-bit, not approximately: multiplying by an array of exact 1.0 is
    # the identity in IEEE 754, and if that ever stops being true here the
    # recorded runs stop being reproducible.
    assert np.array_equal(a, b)


def test_boost_follows_the_onset_not_the_label():
    d = _toy()
    b = onset_proximity_boost(d, within_s=5.0, factor=3.0)
    for w, got in zip(d.windows, b):
        after_onset = 0.0 < (w.end_sec - 100.0) <= 5.0
        assert got == (3.0 if after_onset else 1.0), w.end_sec
    # A window ending exactly at the onset holds no ictal signal.
    at_onset = next(i for i, w in enumerate(d.windows) if w.end_sec == 100.0)
    assert b[at_onset] == 1.0
    # A window deep inside the seizure is ictal but not early, so it is not
    # boosted -- that is the whole distinction from class weighting.
    deep = next(i for i, w in enumerate(d.windows) if w.end_sec == 115.0)
    assert d.windows[deep].label == 1 and b[deep] == 1.0


def test_boost_multiplies_the_class_weights_rather_than_replacing_them():
    d = _toy()
    plain = np.asarray(make_class_balanced_sampler(d).weights, dtype=np.float64)
    boosted = np.asarray(make_class_balanced_sampler(
        d, onset_within_s=5.0, onset_factor=3.0).weights, dtype=np.float64)
    ratio = boosted / plain
    assert set(np.unique(np.round(ratio, 12))) == {1.0, 3.0}


def test_a_non_positive_factor_is_refused():
    d = _toy()
    for bad in (0.0, -1.0):
        try:
            onset_proximity_boost(d, within_s=5.0, factor=bad)
        except ValueError:
            continue
        raise AssertionError(f"factor={bad} should have been refused")
