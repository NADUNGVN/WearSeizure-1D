#!/usr/bin/env bash
# Phase 5 -- lever L3: distil a multi-channel teacher into the single-channel student.
#
# Why this is the last idea worth trying
# -------------------------------------
# Four levers have been measured since cohort pre-training (L1) produced the
# only gain on record, +6 to +9pp:
#   L5, corpus at four electrode positions   null
#   L5, corpus at one position               significantly WORSE
#   L4, checkpoint selection on AUPRC        null for k5only; worse FAR for frontiers2d
# So more data does not help and a better selection criterion does not help. L3
# is the remaining hypothesis: that the single channel is short of INFORMATION,
# and that a teacher reading every channel of the same recordings has something
# to transfer.
#
# The smoke test says it might. On chb01's fold the multi-channel teacher reached
# validation AUPRC 0.99 where the student reached 0.75 -- the largest capability
# gap measured anywhere in this project.
#
# The three arms
# --------------
#   control      already on disk (rows 32-33), not re-run here
#   L3           teacher reads every channel
#   L3single     SAME teacher architecture, fed the student's one channel
#
# The third arm is not optional. The teacher is both multi-channel AND much
# wider than the student, so a gain from the second arm alone confounds "more
# channels" with "more capacity" -- the first thing a reviewer will ask. Only
# the difference between arms two and three isolates the channels.
#
# Cost note: pre-training is shared with the control
# --------------------------------------------------
# L3 changes the fine-tuning loss and nothing else, so its cohort
# initialisations are bit-identical to the control's.
# `train.pretrain.share_cache_with_control=true` reuses them instead of spending
# about forty hours reproducing what is already on disk. That flag is wrong for
# a lever that changes pre-training -- L4 correctly rebuilt its own -- and is
# set here only because this one does not.
#
# Teachers are cached per fold and shared across student architectures within an
# arm, so the second architecture pays only for its students.
#
#   nohup bash scripts/run_phase5_l3.sh > phase5.out 2>&1 &
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
LOG="$ART/phase5_l3.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
MODELS=(wearseizure1d_k5only baseline_frontiers2d)
SEEDS=(0 1 2)

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 5 start. commit=$(git rev-parse --short HEAD) ulimit -n=$(ulimit -n)"

# The control's cohort initialisations must already exist, or
# share_cache_with_control silently trains them here and the "shared" claim is
# false. Checked rather than assumed.
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    n=$(ls "$ART/pretrain/$model/$WINDOW/seed$seed"/*.pt 2>/dev/null | wc -l)
    [ "$n" -eq 13 ] || { say "ABORT: control cohort inits for $model seed=$seed are $n/13"; exit 1; }
  done
done
say "control cohort initialisations present for all 6 combinations"

for arm in L3 L3single; do
  if [ "$arm" = "L3single" ]; then
    ARM_ARGS="train.distill.single_channel_teacher=true"
  else
    ARM_ARGS="train.distill.single_channel_teacher=false"
  fi

  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      say "train: $model seed=$seed arm=$arm"
      run python scripts/train.py profile=server data=chbmit \
        model="$model" seed="$seed" \
        train.pretrain.enabled=true train.pretrain.share_cache_with_control=true \
        train.distill.enabled=true $ARM_ARGS train.run_tag="$arm"

      n=$(ls "$ART/$model/$SPLIT/${WINDOW}__${arm}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
      say "folds $model seed=$seed arm=$arm: $n/66"
      # A partial cohort produced sensitivity 1.0000 in the first L4 attempt --
      # a number that looks like a triumph and covers three easy patients.
      [ "$n" -eq 66 ] || { say "ABORT: $model seed=$seed arm=$arm has $n/66 folds"; exit 1; }

      run python scripts/rethreshold.py profile=server data=chbmit \
        model="$model" seed="$seed" train.run_tag="$arm" postprocess=hysteresis_widegrid
    done

    say "evaluate: $model arm=$arm, 3 seeds, v2 gates"
    run python scripts/evaluate.py profile=server data=chbmit model="$model" \
      train.run_tag="$arm" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
      profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml

    say "paired bootstrap: $model $arm vs its control"
    run python scripts/paired_bootstrap.py \
      "$ART/$model/$SPLIT/${WINDOW}__${arm}" "$ART/$model/$SPLIT/$WINDOW" \
      --all-metrics --json "$ART/paired_${model}_${arm}_vs_control.json"
  done
done

# The comparison the third arm exists for: multi-channel teacher against the
# same teacher restricted to one channel. Anything the two share is capacity;
# only what separates them is the channels.
for model in "${MODELS[@]}"; do
  say "paired bootstrap: $model L3 (multi-channel) vs L3single (same teacher, one channel)"
  run python scripts/paired_bootstrap.py \
    "$ART/$model/$SPLIT/${WINDOW}__L3" "$ART/$model/$SPLIT/${WINDOW}__L3single" \
    --all-metrics --json "$ART/paired_${model}_L3_vs_L3single.json"
done

say "Phase 5 done. Send back $LOG and the paired_*_L3*.json files"
