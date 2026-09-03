#!/usr/bin/env bash
# Does the window stride explain the rest of the gap to the published number?
#
# What is settled and what is not
# -------------------------------
# A7 measured, with one body of code on one body of data, that the published
# split rule inflates window sensitivity from 0.6033 to 0.9229 -- 31 points.
# That part is a measurement and no longer an inference.
#
# It did NOT reach the published 0.9962. The reproduction kept this project's
# 1-second stride; Chung 2024 slides by ONE SAMPLE, 1/256 s. A 57-second seizure
# gives them about 14,600 ictal windows where it gives us about 54, so their
# minority class is ~256x larger at 0.6% prevalence. That is the leading
# explanation for the remaining 7.3 points, and it is testable.
#
# Why a sweep and not the real thing
# ----------------------------------
# A 1-sample stride is ~37 million windows per fold. The sweep asks the same
# question affordably: if sensitivity climbs monotonically as the stride falls,
# the stride is the explanation and the trend can be reported as such. If it
# plateaus near 0.92, something else is going on and the reproduction stays
# incomplete for a different reason -- which is worth knowing either way.
#
# 13 folds, one per patient. `train.max_folds` takes the first n folds, which
# are all chb01; this spans the cohort, noisier than 66 but comparable in kind.
#
#   STRIDES="0.5 0.25 0.125" bash scripts/run_stride_sweep.sh
set -uo pipefail

missing=""
[ -z "${CHBMIT_RAW_DIR:-}" ] && missing="$missing CHBMIT_RAW_DIR"
[ -z "${WEARSEIZURE_ARTIFACTS_DIR:-}" ] && missing="$missing WEARSEIZURE_ARTIFACTS_DIR"
if [ -n "$missing" ]; then
  echo "ABORT: not set:$missing"
  echo "  export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0"
  echo "  export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts"
  exit 1
fi
ulimit -n 65536 2>/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WEARSEIZURE_RUN_HOST="$(hostname)"

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/stride_sweep_$(hostname).log"
MODEL="${MODEL:-wearseizure1d_k5only}"
STRIDES="${STRIDES:-0.5 0.25 0.125}"
WORKERS="${WORKERS:-0}"
BATCH="${BATCH:-1024}"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "stride sweep. model=$MODEL strides=$STRIDES host=$(hostname) commit=$(git rev-parse --short HEAD)"

for stride in $STRIDES; do
  say "stride=${stride}s"
  # Rung A only: the sweep is about reproducing the published protocol, and the
  # honest rungs are already measured at the project's own 1s stride.
  python scripts/run_leaky_repro.py profile=server data=chbmit \
    model="$MODEL" +rung=A_as_published +fold_subset=one_per_patient \
    window.stride_s="$stride" profile.num_workers="$WORKERS" train.batch_size="$BATCH" \
    2>&1 | tee -a "$LOG"

  tag="stride$(echo "$stride" | tr '.' 'p')s"
  n=$(ls "$ART/leaky_repro/A_as_published/$MODEL/$tag"/*.json 2>/dev/null | wc -l)
  say "stride=${stride}s: $n/13 folds"
  [ "$n" -eq 13 ] || { say "ABORT: stride=$stride produced $n/13 folds"; exit 1; }
done

say "sweep done. Compare against the 1s row, which is the 66-fold table:"
say "  python scripts/summarise_leaky_repro.py \$WEARSEIZURE_ARTIFACTS_DIR --markdown"
say "NOTE: sweep rows are 13 folds, the 1s row is 66. Do not read a small"
say "      difference between them as a stride effect."
