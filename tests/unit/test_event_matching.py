from __future__ import annotations

from wearseizure.eval.event_matching import match_events_to_alarms
from wearseizure.eval.metrics_event import compute_event_metrics


def test_match_events_to_alarms_basic_match_miss_false_alarm():
    events = [("e0", 10.0, 20.0), ("e1", 100.0, 110.0)]
    alarms = [(12.0, 18.0), (200.0, 205.0)]  # matches e0, misses e1, one false alarm
    result = match_events_to_alarms(events, alarms)
    assert [m[0] for m in result.matched] == ["e0"]
    assert result.missed_event_ids == ["e1"]
    assert result.false_alarms == [(200.0, 205.0)]


def test_each_alarm_used_at_most_once():
    events = [("e0", 10.0, 20.0), ("e1", 15.0, 25.0)]  # overlapping events
    alarms = [(12.0, 22.0)]  # a single alarm overlapping both
    result = match_events_to_alarms(events, alarms)
    assert len(result.matched) == 1
    assert len(result.missed_event_ids) == 1
    assert result.false_alarms == []


def test_compute_event_metrics_sensitivity_and_far():
    events = [("e0", 10.0, 20.0), ("e1", 100.0, 110.0)]
    alarms = [(12.0, 18.0), (500.0, 505.0)]
    metrics = compute_event_metrics(events, alarms, exposure_hours=2.0)
    assert metrics.n_events == 2
    assert metrics.n_matched == 1
    assert metrics.sensitivity == 0.5
    assert metrics.n_false_alarms == 1
    assert metrics.far_per_hour == 0.5
    assert metrics.delays_s == [2.0]  # alarm starts at 12, event onset at 10


def test_compute_event_metrics_delay_clipped_at_zero_for_early_alarm():
    events = [("e0", 10.0, 20.0)]
    alarms = [(9.0, 15.0)]  # alarm starts before onset
    metrics = compute_event_metrics(events, alarms, exposure_hours=1.0)
    assert metrics.delays_s == [0.0]


def test_compute_event_metrics_handles_zero_events():
    metrics = compute_event_metrics([], [(1.0, 2.0)], exposure_hours=1.0)
    assert metrics.n_events == 0
    import math

    assert math.isnan(metrics.sensitivity)
    assert metrics.n_false_alarms == 1
