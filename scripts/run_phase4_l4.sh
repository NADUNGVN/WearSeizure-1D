#!/usr/bin/env bash
# Phase 4 -- lever L4: select checkpoints on validation AUPRC, not cross-entropy.
#
# Why a script and not a pasted loop
# ----------------------------------
# The first L4 attempt was a hand-typed block and it died on "Too many open
# files" at the fourth cohort initialisation of every one of the six
# combinations -- because that block omitted `ulimit -n 65536`. Every phase that
# ran through a script had it; the one that was retyped did not.
# docs/SERVER_INVENTORY.md documents that failure twice already, and it is not a
# thing to rely on remembering.
#
# What it measures
# ----------------
# The paper's metric is event sensitivity at a bounded false-alarm rate.
# Cross-entropy is a poor stand-in at this imbalance: ictal windows are ~0.5% of
# the data, so the loss can fall while the model gets no better at the minority
# class that decides every reported number. AUPRC scores the positive class
# specifically and ignores the true-negative mass, and it is threshold-free --
# selecting on event sensitivity directly would need thresholds that
# threshold_selection.py fits on the same validation partition afterwards.
#
# One caveat to read the result with: the first attempt's partial log showed
# folds reaching epochs 57-59, i.e. the 60-epoch ceiling rather than early
# stopping. AUPRC keeps improving after cross-entropy has saturated, so L4 also
# trains LONGER. If the finished run shows many folds at the ceiling, "selected
# on AUPRC" and "trained longer" are confounded and the ceiling has to be raised
# before the comparison means anything. The script counts them and says so.
#
# Resumable: completed folds and cached initialisations are skipped.
#
#   nohup bash scripts/run_phase4_l4.sh > phase4.out 2>&1 &
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
LOG="$ART/phase4_l4.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
TAG="${TAG:-L4}"
MODELS=(wearseizure1d_k5only baseline_frontiers2d)
SEEDS=(0 1 2)

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 4 start. commit=$(git rev-parse --short HEAD) ulimit -n=$(ulimit -n) tag=$TAG"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    say "train: $model seed=$seed (model_selection=val_auprc)"
    run python scripts/train.py profile=server data=chbmit \
      model="$model" seed="$seed" train.pretrain.enabled=true \
      train.model_selection=val_auprc train.run_tag="$TAG"

    n=$(ls "$ART/$model/$SPLIT/${WINDOW}__${TAG}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
    say "folds $model seed=$seed: $n/66"
    # Stop rather than build a comparison on a partial cohort. The first attempt
    # produced 13 folds covering 3 patients and reported sensitivity 1.0000 --
    # a number that looks like a triumph and means nothing.
    [ "$n" -eq 66 ] || { say "ABORT: $model seed=$seed has $n/66 folds"; exit 1; }

    run python scripts/rethreshold.py profile=server data=chbmit \
      model="$model" seed="$seed" train.run_tag="$TAG" postprocess=hysteresis_widegrid
  done

  say "evaluate: $model, 3 seeds, v2 gates"
  run python scripts/evaluate.py profile=server data=chbmit model="$model" \
    train.run_tag="$TAG" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
    profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml

  say "paired bootstrap: $model with $TAG vs its control"
  run python scripts/paired_bootstrap.py \
    "$ART/$model/$SPLIT/${WINDOW}__${TAG}" "$ART/$model/$SPLIT/$WINDOW" \
    --all-metrics --json "$ART/paired_${model}_${TAG}_vs_control.json"
done

# The confound check, reported rather than left for someone to notice.
ceiling=$(grep -c "epoch 59:" "$LOG" 2>/dev/null || echo 0)
total=$(grep -c "early stopping at epoch" "$LOG" 2>/dev/null || echo 0)
say "folds reaching the 60-epoch ceiling: $ceiling (early-stopped: $total)"
if [ "$ceiling" -gt 0 ]; then
  say "NOTE: folds hit the epoch ceiling, so 'selected on AUPRC' and 'trained longer'"
  say "      are partly confounded. Raise train.epochs and re-run before claiming L4"
  say "      is what moved the numbers."
fi

say "Phase 4 done. Send back $LOG and paired_*_${TAG}_vs_control.json"
