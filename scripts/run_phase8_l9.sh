#!/usr/bin/env bash
# Phase 8 -- lever L9: onset-proximity sampling, against L8 on the same folds.
#
# WHAT THIS TESTS. On the handover vectors every window the detector gets wrong
# is a false negative in the first eighteen seconds after onset, and there is
# not one false positive: it reacts late rather than misclassifying. Of the
# twenty-four second delay on chb15_40, four seconds is the window floor, two
# are run_length, and roughly fifteen are the model. This arm raises the
# sampling probability of windows in the first `WITHIN` seconds of a seizure by
# `FACTOR`, so that being slow costs the model something during training.
#
# WHAT WOULD MAKE IT FAIL. Those early windows are three-quarters pre-ictal
# signal. Teaching the model to fire on them may teach it to fire on pre-ictal
# signal that never becomes a seizure, which costs false alarms. A delay
# improvement bought with sensitivity or FAR is not an improvement -- the
# summariser reports all three and reads the worst case the interval allows.
#
# This is L8's recipe plus the lever and nothing else, on seed 0, so the
# comparison against the existing L8 seed-0 run is paired fold for fold. The
# teacher checkpoints and the pre-training cache already exist from Phase 6;
# this arm reuses both rather than rebuilding them.
set -uo pipefail

: "${WEARSEIZURE_ARTIFACTS_DIR:?set it, or put it in .env at the repo root}"
: "${CHBMIT_RAW_DIR:?set it, or put it in .env at the repo root}"

# The open-file limit killed the first L4 attempt during cohort init. Same
# recipe here, same risk.
if ! ulimit -n 65536 2>/dev/null && [ "$(ulimit -n)" -lt 8192 ]; then
  echo "ABORT: open-file limit is $(ulimit -n); raise it or put 'ulimit -n 65536' in ~/.bashrc."
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase8_l9.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
TEACHER=baseline_frontiers2d
STUDENT=wearseizure1d_k5only
BASE_TAG="${BASE_TAG:-L8}"
TAG="${TAG:-L9}"
SEEDS=(${SEEDS:-0})
ALPHA="${ALPHA:-0.5}"
WITHIN="${WITHIN:-10}"
FACTOR="${FACTOR:-3}"
WORKERS="${WORKERS:-}"
WORKER_ARG=()
[ -n "$WORKERS" ] && WORKER_ARG=(profile.num_workers="$WORKERS")

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 8 start. commit=$(git rev-parse --short HEAD) tag=$TAG within=${WITHIN}s factor=$FACTOR seeds=${SEEDS[*]}"

# Both preconditions up front. Discovering either at fold 40 wastes a day, and
# discovering the second one never is worse: a baseline arm that turns out to
# be missing folds makes the whole comparison quietly partial.
for seed in "${SEEDS[@]}"; do
  n=$(ls "$ART/$TEACHER/$SPLIT/$WINDOW/seed$seed"/*.pt 2>/dev/null | wc -l)
  [ "$n" -eq 66 ] || { say "ABORT: teacher checkpoints for seed=$seed are $n/66"; exit 1; }
  b=$(ls "$ART/$STUDENT/$SPLIT/${WINDOW}__${BASE_TAG}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
  [ "$b" -eq 66 ] || { say "ABORT: baseline $BASE_TAG seed=$seed has $b/66 folds; there is nothing to compare against"; exit 1; }
done
say "preconditions met: teacher 66/66, baseline $BASE_TAG 66/66"

for seed in "${SEEDS[@]}"; do
  say "train: $STUDENT seed=$seed, L8 recipe + onset weighting (within=${WITHIN}s factor=$FACTOR)"
  run python scripts/train.py profile=server data=chbmit \
    model="$STUDENT" seed="$seed" \
    train.pretrain.enabled=true train.pretrain.share_cache_with_control=true \
    train.distill.enabled=true train.distill.teacher_model="$TEACHER" \
    train.distill.alpha="$ALPHA" \
    train.onset_weight.within_s="$WITHIN" train.onset_weight.factor="$FACTOR" \
    train.run_tag="$TAG" "${WORKER_ARG[@]}"

  n=$(ls "$ART/$STUDENT/$SPLIT/${WINDOW}__${TAG}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
  say "folds $STUDENT seed=$seed: $n/66"
  if [ "$n" -ne 66 ]; then
    say "INCOMPLETE: $n/66. A partial cohort reported sensitivity 1.0000 once before -- a"
    say "number that looked like a triumph and covered three easy patients. Do not read"
    say "the summary below until this is 66."
  fi
done

say "comparison"
run python scripts/summarise_onset_weight.py "$ART" --baseline "$BASE_TAG" --arm "$TAG" --seed "${SEEDS[0]}"

say "Phase 8 done."
