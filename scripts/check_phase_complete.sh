#!/usr/bin/env bash
# Did Phase 5 (L3) and Phase 6 (L8) actually finish, or did they abort?
#
# Both phase scripts exit on a short fold count rather than build a comparison
# on a partial cohort -- which is right, but it means an aborted run and a
# finished run look identical from the outside: no process, a log, some files.
# Twice already a run reported as "done" had aborted or was still going.
#
# This checks the three things that distinguish them: the closing line of the
# log, the fold count of EVERY expected combination, and whether the paired
# bootstrap outputs the phase exists to produce are actually on disk.
#
#   bash scripts/check_phase_complete.sh
set -uo pipefail

ART="${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
bad=0

folds() {  # model tag seed
  local d="$ART/$1/$SPLIT/${WINDOW}${2:+__$2}/seed$3"
  ls "$d"/*.metrics.json 2>/dev/null | wc -l
}

check() {  # model tag seed
  local n; n=$(folds "$1" "$2" "$3")
  if [ "$n" -eq 66 ]; then
    printf '  OK    %-24s %-10s seed%s  %s/66\n' "$1" "${2:-<control>}" "$3" "$n"
  else
    printf '  SHORT %-24s %-10s seed%s  %s/66\n' "$1" "${2:-<control>}" "$3" "$n"
    bad=1
  fi
}

banner() {  # log-path phase-name
  echo
  echo "=== $2 ==="
  if [ ! -f "$1" ]; then echo "  no log at $1 -- this phase never started"; bad=1; return; fi
  # Only aborts from the CURRENT attempt count. Logs are appended across
  # restarts, so a fixed-and-rerun phase still carries the abort line that made
  # it get fixed -- reporting that as a failure is how this script called a
  # healthy Phase 5 "ABORTED" on its first use.
  local start; start=$(grep -n "Phase .* start\." "$1" | tail -1 | cut -d: -f1)
  local aborts; aborts=$(tail -n "+${start:-1}" "$1" | grep -n "ABORT")
  if [ -n "$aborts" ]; then
    echo "  ABORTED since the last start (line $start):"
    echo "$aborts" | sed 's/^/    /'
    bad=1
  elif grep -q "ABORT" "$1"; then
    echo "  note: aborts exist from an EARLIER attempt, before line $start -- not counted"
  fi
  echo "  last line: $(tail -1 "$1")"
}

banner "$ART/phase5_l3.log" "Phase 5 -- lever L3 (multi-channel teacher)"
for arm in L3 L3single; do
  for model in wearseizure1d_k5only baseline_frontiers2d; do
    for seed in 0 1 2; do check "$model" "$arm" "$seed"; done
  done
done

banner "$ART/phase6_l8.log" "Phase 6 -- lever L8 (frontiers2d -> k5only)"
for seed in 0 1 2; do check wearseizure1d_k5only L8 "$seed"; done

echo
echo "=== paired bootstrap outputs ==="
for f in \
  paired_wearseizure1d_k5only_L3_vs_control.json \
  paired_baseline_frontiers2d_L3_vs_control.json \
  paired_wearseizure1d_k5only_L3single_vs_control.json \
  paired_baseline_frontiers2d_L3single_vs_control.json \
  paired_wearseizure1d_k5only_L3_vs_L3single.json \
  paired_baseline_frontiers2d_L3_vs_L3single.json \
  paired_wearseizure1d_k5only_L8_vs_control.json \
  paired_wearseizure1d_k5only_L8_vs_teacher.json ; do
  if [ -s "$ART/$f" ]; then printf '  OK      %s\n' "$f"; else printf '  MISSING %s\n' "$f"; bad=1; fi
done

echo
if [ "$bad" -eq 0 ]; then
  echo "BOTH PHASES COMPLETE. Send the log tails and the paired_*.json files."
else
  echo "NOT COMPLETE -- see SHORT / ABORTED / MISSING above. Do not report these as results."
fi
exit "$bad"
