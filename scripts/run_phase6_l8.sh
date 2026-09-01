#!/usr/bin/env bash
# Phase 6 -- lever L8: distil the finished `frontiers2d` run into `k5only`.
#
# The gap this exists to close
# ---------------------------
# Rows 32-33, three seeds each, identical recipe, identical single-channel
# input:
#
#   baseline_frontiers2d   sens 0.9726 +/- 0.0037   2,523,328 MACs
#   wearseizure1d_k5only   sens 0.9358 +/- 0.0304     585,920 MACs
#
# The architecture that meets the MAC gate misses the 0.95 sensitivity gate; the
# one that meets sensitivity exceeds the MAC gate 4.3x. That is the whole of
# what still blocks the paper, and four measured levers have not moved it: L5
# was negative at both corpus widths, L4 moved the operating point without
# raising quality.
#
# 3.7pp between two models reading exactly the same data is a statement about
# CAPACITY, not about information. L8 tests whether it is real capacity or just
# a harder optimisation problem: if the small student can be taught to imitate
# the large one's outputs, the gap was optimisation and the paper gets
# "0.97 at 4.3x fewer MACs". If it cannot, 585,920 MACs is genuinely too small
# and the PE array has to be sized larger -- which is far cheaper to learn now
# than after there is RTL.
#
# Why the teacher is not trained here
# ----------------------------------
# The number worth distilling is 0.9726, and that number depends on cohort
# pre-training (lever L1). A frontiers2d teacher retrained per fold from scratch
# reaches about 0.88. So the teacher IS row 33: its saved checkpoints, loaded
# and scored. Nothing is retrained, and the teacher is by construction the exact
# model that produced the published-in-the-log figure.
#
# Teacher and student are paired by seed (seed 0 distils from seed 0), so the
# three runs stay three independent replicates rather than three students of one
# teacher.
#
#   nohup bash scripts/run_phase6_l8.sh > phase6.out 2>&1 &
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"
[ -d "$CHBMIT_RAW_DIR" ] || { echo "CHBMIT_RAW_DIR does not exist: $CHBMIT_RAW_DIR"; exit 1; }

ulimit -n 65536
if [ "$(ulimit -n)" -lt 65536 ]; then
  echo "ABORT: could not raise the open-file limit past $(ulimit -n)."
  echo "This is what killed the first L4 attempt at the fourth cohort init of every"
  echo "combination. Raise the hard limit, or put 'ulimit -n 65536' in ~/.bashrc."
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase6_l8.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
TEACHER=baseline_frontiers2d
STUDENT=wearseizure1d_k5only
TAG="${TAG:-L8}"
SEEDS=(0 1 2)
ALPHA="${ALPHA:-0.5}"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 6 start. commit=$(git rev-parse --short HEAD) ulimit -n=$(ulimit -n) tag=$TAG alpha=$ALPHA"

# Every teacher checkpoint must exist BEFORE anything trains. train.py refuses a
# fold whose teacher is missing rather than quietly training it undistilled, but
# discovering that at fold 40 wastes a day -- so it is checked up front.
for seed in "${SEEDS[@]}"; do
  n=$(ls "$ART/$TEACHER/$SPLIT/$WINDOW/seed$seed"/*.pt 2>/dev/null | wc -l)
  [ "$n" -eq 66 ] || { say "ABORT: teacher checkpoints for seed=$seed are $n/66 (need row 33 complete)"; exit 1; }
done
say "teacher checkpoints present: 66 folds x 3 seeds"

for seed in "${SEEDS[@]}"; do
  say "train: $STUDENT seed=$seed distilling from $TEACHER seed=$seed"
  run python scripts/train.py profile=server data=chbmit \
    model="$STUDENT" seed="$seed" \
    train.pretrain.enabled=true train.pretrain.share_cache_with_control=true \
    train.distill.enabled=true train.distill.teacher_model="$TEACHER" \
    train.distill.alpha="$ALPHA" train.run_tag="$TAG"

  n=$(ls "$ART/$STUDENT/$SPLIT/${WINDOW}__${TAG}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
  say "folds $STUDENT seed=$seed: $n/66"
  # A partial cohort reported sensitivity 1.0000 in the first L4 attempt -- a
  # number that looks like a triumph and covers three easy patients.
  [ "$n" -eq 66 ] || { say "ABORT: $STUDENT seed=$seed has $n/66 folds"; exit 1; }

  run python scripts/rethreshold.py profile=server data=chbmit \
    model="$STUDENT" seed="$seed" train.run_tag="$TAG" postprocess=hysteresis_widegrid
done

say "evaluate: $STUDENT arm=$TAG, 3 seeds, v2 gates"
run python scripts/evaluate.py profile=server data=chbmit model="$STUDENT" \
  train.run_tag="$TAG" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
  profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml

# The comparison L8 exists for: did the student move toward its teacher?
say "paired bootstrap: $STUDENT $TAG vs its own control (did distillation help?)"
run python scripts/paired_bootstrap.py \
  "$ART/$STUDENT/$SPLIT/${WINDOW}__${TAG}" "$ART/$STUDENT/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_${STUDENT}_${TAG}_vs_control.json"

say "paired bootstrap: $STUDENT $TAG vs the TEACHER (did it close the gap?)"
run python scripts/paired_bootstrap.py \
  "$ART/$STUDENT/$SPLIT/${WINDOW}__${TAG}" "$ART/$TEACHER/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_${STUDENT}_${TAG}_vs_teacher.json"

say "Phase 6 done. Send back $LOG and both paired_*_${TAG}_*.json files"
say "Read them together: vs_control says whether distillation did anything,"
say "vs_teacher says whether what it did was enough to reach 0.9726."
