"""Alarm <-> ground-truth event matching (memo 5.2).

Each alarm is used for at most one event (sweep in onset order, first
unused overlapping alarm wins), so counts can never double-count a single
alarm as covering two events.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchResult:
    matched: list[tuple[str, tuple[float, float], tuple[float, float]]]  # (event_id, event_interval, alarm_interval)
    missed_event_ids: list[str]
    false_alarms: list[tuple[float, float]]


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def match_events_to_alarms(
    events: list[tuple[str, float, float]],
    alarms: list[tuple[float, float]],
) -> MatchResult:
    events_sorted = sorted(events, key=lambda e: e[1])
    used = [False] * len(alarms)
    matched: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    missed_ids: list[str] = []

    for event_id, onset, offset in events_sorted:
        found_idx = None
        for i, (a_start, a_end) in enumerate(alarms):
            if used[i]:
                continue
            if _overlaps(onset, offset, a_start, a_end):
                found_idx = i
                break
        if found_idx is not None:
            used[found_idx] = True
            matched.append((event_id, (onset, offset), alarms[found_idx]))
        else:
            missed_ids.append(event_id)

    false_alarms = [a for i, a in enumerate(alarms) if not used[i]]
    return MatchResult(matched=matched, missed_event_ids=missed_ids, false_alarms=false_alarms)
