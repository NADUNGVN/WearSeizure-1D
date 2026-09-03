#!/usr/bin/env bash
# Phase 7 -- the capacity ladder. The first arm that changes model SIZE.
#
# Why capacity, and why now
# -------------------------
# Five levers have been measured and none moved sensitivity:
#   L1  cohort pre-training      the only gain on record, +6 to +9pp
#   L5  wider pre-training corpus   null, and significantly worse at one position
#   L4  AUPRC checkpoint selection  operating point only
#   L3  multi-channel teacher       significantly WORSE FAR for frontiers2d
#   L8  same-input teacher          +1.3pp macro, +2.2pp micro, not significant
#
# L8 is the informative one. A stronger teacher reading the student's own
# channel did move it, and still left it significantly behind that teacher
# (CI [-4.60, -0.51]pp). That is what a capacity limit looks like: the student
# can be pushed toward the target and cannot arrive. Nothing tried so far has
# changed the student's size.
#
# Where the budget comes from
# --------------------------
# Not from relaxing the gate. `context` is two depthwise-separable k5 convs at
# dilation 8 and 16, running on a 32-sample sequence -- a k5 at dilation 16
# spans 65 samples, so most of its taps read padding. Narrowing it 64 -> 16
# frees 218,256 MACs, 37% of the model, from the layer least able to use them,
# and shrinks the largest streaming line buffer in the design.
#
# Spent on 50% wider stages, the result is 626,736 MACs against the M10 gate of
# 630,832 -- and 9,414 parameters against the baseline's 11,786.
#
# One variable per rung
# ---------------------
#   k5only        ctx 64, stages (16,24,32,48)   585,920 MACs   (control, row 32)
#   k5only_ctx16  ctx 16, stages (16,24,32,48)   367,664 MACs   isolates context
#   k5only_wide   ctx 16, stages (24,36,48,72)   626,736 MACs   isolates stages
#
# `wide` against the control changes two things at once. It is only
# interpretable through `ctx16`, which is why the cheap rung is not optional:
# comparing the two ends directly is the same confound that invalidated the
# row-15 architecture claim.
#
# A null on rung 1 is a result by itself: 367,664 MACs is 6.9x fewer than
# baseline_frontiers2d, against 4.3x for the current k5only.
#
#   WORKERS=7 nohup bash scripts/run_phase7_capacity.sh > phase7.out 2>&1 &
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
LOG="$ART/phase7_capacity.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
CONTROL=wearseizure1d_k5only
MODELS=(wearseizure1d_k5only_ctx16 wearseizure1d_k5only_wide)
# Seeds are independent of one another, so a host with spare time can take some.
# SEEDS="1" on one machine and SEEDS="2" on another halves the wall clock, and
# each seed writes to its own seed<N> directory so there is nothing to collide
# on even across the shared NFS artifacts tree.
#
# A host running a SUBSET must not run the evaluate/bootstrap tail: those pool
# all three seeds and would report a two-seed result as if it were three.
# The script detects that and stops after training, saying so.
SEEDS=(${SEEDS:-0 1 2})
FULL_RUN=1
[ "${SEEDS[*]}" = "0 1 2" ] || FULL_RUN=0
WORKERS="${WORKERS:-}"
WORKER_ARG=()
[ -n "$WORKERS" ] && WORKER_ARG=(profile.num_workers="$WORKERS")
export WEARSEIZURE_RUN_HOST="$(hostname)"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 7 start. commit=$(git rev-parse --short HEAD) ulimit -n=$(ulimit -n) workers=${WORKERS:-<profile default>}"

# These are NEW architectures, so their cohort initialisations do not exist and
# must NOT be shared with the control -- lever L1 pre-trains the same network
# that is then fine-tuned. share_cache_with_control is deliberately absent here;
# it is correct only for a lever that leaves pre-training unchanged, as L3 and
# L8 do. Each model therefore builds its own 13 initialisations per seed first.
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    say "train: $model seed=$seed (own cohort pre-training)"
    run python scripts/train.py profile=server data=chbmit \
      model="$model" seed="$seed" train.pretrain.enabled=true "${WORKER_ARG[@]}"

    n=$(ls "$ART/$model/$SPLIT/$WINDOW/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
    say "folds $model seed=$seed: $n/66"
    [ "$n" -eq 66 ] || { say "ABORT: $model seed=$seed has $n/66 folds"; exit 1; }

    run python scripts/rethreshold.py profile=server data=chbmit \
      model="$model" seed="$seed" postprocess=hysteresis_widegrid
  done

  if [ "$FULL_RUN" -eq 0 ]; then
    say "SEEDS=${SEEDS[*]} is a subset -- skipping evaluate/bootstrap for $model."
    say "Run this script with the default SEEDS on one host once every seed exists."
    continue
  fi
  say "evaluate: $model, 3 seeds, v2 gates"
  run python scripts/evaluate.py profile=server data=chbmit model="$model" \
    postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
    profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml
done

if [ "$FULL_RUN" -eq 0 ]; then
  say "subset run finished training. The comparisons need all three seeds:"
  say "  bash scripts/run_phase7_capacity.sh   # with default SEEDS, once seeds 0-2 are on disk"
  exit 0
fi

# Rung 1: does the context block's width earn its 37% of the MACs?
say "paired bootstrap: ctx16 vs control (isolates the context width)"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only_ctx16/$SPLIT/$WINDOW" "$ART/$CONTROL/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_ctx16_vs_control.json"

# Rung 2: does spending those MACs on width buy sensitivity?
say "paired bootstrap: wide vs ctx16 (isolates the stage width)"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only_wide/$SPLIT/$WINDOW" "$ART/wearseizure1d_k5only_ctx16/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_wide_vs_ctx16.json"

# The headline, reported but NOT to be read on its own: two variables move.
say "paired bootstrap: wide vs control (headline; interpret only via the two above)"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only_wide/$SPLIT/$WINDOW" "$ART/$CONTROL/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_wide_vs_control.json"

# And against the model that has been out of reach since Phase 2.
say "paired bootstrap: wide vs baseline_frontiers2d (the 0.9726 target)"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only_wide/$SPLIT/$WINDOW" "$ART/baseline_frontiers2d/$SPLIT/$WINDOW" \
  --all-metrics --json "$ART/paired_wide_vs_frontiers2d.json"

say "Phase 7 done. Send back $LOG and the four paired_*.json files."
say "Read in order: ctx16_vs_control, then wide_vs_ctx16, then the two headlines."
