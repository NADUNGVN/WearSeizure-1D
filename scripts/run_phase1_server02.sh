#!/usr/bin/env bash
# Phase 1 on SERVER-02 alone: three seeds x two architectures, ~50h sequential.
#
# What it answers
# ---------------
#   1. Is row 24 (0.9359) really better than row 22 (0.9218)? They are 1.4pp
#      apart on 77 seizures -- about one seizure -- and no run on record has an
#      error bar, so today the honest answer is "unknown".
#   2. wearseizure1d or wearseizure1d_k5only? RESEARCH_REALITY_CHECK section
#      11.4 records the architecture as settled on k5only, but every row from
#      21 to 26 was run on the default. L1 + k5only has never been tried, and
#      only k5only meets the MAC gate's target tier (585,920 vs 765,632), which
#      is what the paper's compute claim rests on.
#
# Resumable by design. Every stage skips work that already exists:
# get_or_train_cohort_init reuses a cached init whose corpus hash matches, and
# train.py skips any fold that already has a metrics.json. Re-running after an
# interruption costs a few seconds of scanning, not a re-run. So: run it under
# nohup or tmux, and if it dies, just start it again.
#
#   cd ~/Manh/WearSeizure-1D
#   nohup bash scripts/run_phase1_server02.sh > phase1.out 2>&1 &
#   tail -f phase1.out
#
# Everything also lands in $LOG, one file, timestamped.
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

LOG="${WEARSEIZURE_ARTIFACTS_DIR}/phase1_server02.log"
SEEDS=(0 1 2)
MODELS=(wearseizure1d wearseizure1d_k5only)

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 1 start. commit=$(git rev-parse --short HEAD) log=$LOG"

# ---------------------------------------------------------------------------
# Stage 1 -- cohort pre-training inits, then training, per (model, seed).
#
# Ordered so the cheapest useful answer arrives first: after the two seed-0
# combinations there is already a single-seed k5only-vs-default comparison to
# look at, roughly 4h in rather than 50h in. Seeds 1 and 2 then widen it into
# something with an error bar.
#
# Three concurrent shards for pre-training, num_workers=4 each: the binding
# constraint on a ~12k-parameter model is CUDA kernel-launch overhead, not
# arithmetic or data loading, so concurrent processes interleave each other's
# launch gaps where a bigger batch would not help. 3 x 4 = 12 <= 14 physical
# cores; oversubscribing also brings back the "Too many open files" failure.
# Training itself is a single process, so it gets all 14.
# ---------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    say "pre-training inits: model=$model seed=$seed"
    for i in 0 1 2; do
      python scripts/pretrain_cohort.py profile=server data=chbmit \
        model="$model" seed="$seed" +shard="$i" +n_shards=3 profile.num_workers=4 \
        >> "$LOG" 2>&1 &
    done
    wait
    n_init=$(ls "$WEARSEIZURE_ARTIFACTS_DIR/pretrain/$model/w4s_stride1s/seed$seed"/*.pt 2>/dev/null | wc -l)
    say "inits for $model seed=$seed: $n_init/13"
    if [ "$n_init" -ne 13 ]; then
      say "ABORT: expected 13 inits, found $n_init. Check $LOG."
      exit 1
    fi

    say "training 66 folds: model=$model seed=$seed"
    run python scripts/train.py profile=server data=chbmit \
      model="$model" seed="$seed" train.pretrain.enabled=true
    n_folds=$(ls "$WEARSEIZURE_ARTIFACTS_DIR/$model/patient_specific_loso_edf/w4s_stride1s/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
    say "folds for $model seed=$seed: $n_folds/66"
    # Stop rather than spend another forty hours building on a broken chain.
    # Nothing is lost: every completed fold is on disk and skipped on restart.
    if [ "$n_folds" -ne 66 ]; then
      say "ABORT: expected 66 folds, found $n_folds. Fix the cause, then re-run this script."
      exit 1
    fi
  done
done

# ---------------------------------------------------------------------------
# Stage 2 -- score both post-processing configurations.
#
# rethreshold.py overwrites the same *.metrics.json, so the two configurations
# cannot coexist on disk; each is scored in full, and its multi-seed summary is
# copied aside before the next one overwrites it.
#
# enforce_gates=false here only so an expected gate failure does not kill a
# 50-hour unattended script. Every gate level is still computed, logged, and
# written into report.json exactly as before -- nothing is being hidden.
# ---------------------------------------------------------------------------
score() {                       # score <tag> <extra hydra overrides...>
  local tag="$1"; shift
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      say "rethreshold [$tag]: model=$model seed=$seed"
      run python scripts/rethreshold.py profile=server data=chbmit \
        model="$model" seed="$seed" postprocess=hysteresis_widegrid "$@"
    done

    say "evaluate [$tag]: model=$model, 3 seeds, v1 gates"
    run python scripts/evaluate.py profile=server data=chbmit \
      model="$model" postprocess=hysteresis_widegrid "$@" \
      'train.seeds=[0,1,2]' profile.enforce_gates=false

    say "evaluate [$tag]: model=$model, 3 seeds, PROPOSED v2 gates"
    run python scripts/evaluate.py profile=server data=chbmit \
      model="$model" postprocess=hysteresis_widegrid "$@" \
      'train.seeds=[0,1,2]' profile.enforce_gates=false \
      eval.gates_path=configs/eval/gates_v2_proposed.yaml

    local d="$WEARSEIZURE_ARTIFACTS_DIR/$model/patient_specific_loso_edf/w4s_stride1s"
    cp "$d/report_multiseed.json" "$d/report_multiseed_${tag}.json" 2>/dev/null \
      && say "saved $d/report_multiseed_${tag}.json"
  done
}

score row22cfg
score row24cfg postprocess.run_length=2 postprocess.ema_alpha=0.25

# ---------------------------------------------------------------------------
# Stage 3 -- the comparison the whole run exists for.
#
# Point estimates are not evidence at this margin. The cluster is the patient,
# not the seizure: CHB-MIT patients carry 3 to 20+ seizures, so resampling
# seizures would let one patient dominate a replicate.
#
# Reading it: if the CI on sensitivity_macro CONTAINS 0, the two architectures
# are indistinguishable at this dataset's resolution -- and then k5only wins on
# the tiebreak, being 1.3x cheaper in MACs and the only variant that meets the
# MAC gate's target. Only a CI that excludes 0 in the default's favour makes
# this a real trade-off. Lower is better for far_per_hour_micro and
# delay_mean_s, so a NEGATIVE delta favours A there.
# ---------------------------------------------------------------------------
ART="$WEARSEIZURE_ARTIFACTS_DIR"
say "paired bootstrap: k5only vs default (post-processing = row 24 config, as left on disk)"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only/patient_specific_loso_edf/w4s_stride1s" \
  "$ART/wearseizure1d/patient_specific_loso_edf/w4s_stride1s" \
  --all-metrics --json "$ART/paired_k5only_vs_default.json"

say "Phase 1 done. Send back: $LOG, both report_multiseed_row*.json, paired_k5only_vs_default.json"
