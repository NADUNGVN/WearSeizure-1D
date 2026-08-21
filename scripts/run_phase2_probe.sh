#!/usr/bin/env bash
# Phase 2 probe: does the paper have an architecture claim at all?  (~10h)
#
# The question
# -----------
# Phase 1 compared `wearseizure1d_k5only` WITH cohort pre-training (lever L1)
# against reproduced baselines that were trained WITHOUT it:
#
#   k5only + L1, 3 seeds   0.9179 macro
#   frontiers2d, 1 seed    0.8811    <- no L1
#   compact1d_7k, 1 seed   0.8974    <- no L1
#
# That is not an architecture comparison. L1 is a training recipe, and it is the
# single change that produced the largest jump on record (0.8806 -> 0.9218). If
# Chung et al.'s architecture also jumps to ~0.92 once it gets the same recipe,
# then the paper's contribution is the recipe, not the architecture -- a
# different paper, and one a reviewer will ask about.
#
# Rather than spend ~60h running both baselines at three seeds to find out, this
# runs ONE combination -- frontiers2d + L1, seed 0 -- and reads the answer off
# it. frontiers2d is the right probe: it is the direct reproduction of the
# baseline the project is positioned against, and it was the stronger of the two
# on sensitivity before L1 entered the picture.
#
# How to read the result
# ---------------------
#   frontiers2d + L1 stays near 0.88
#       -> L1 is not what separates the architectures. k5only's lead is
#          plausibly real; commit to the full Phase 2 grid and the paper keeps
#          its architecture claim.
#
#   frontiers2d + L1 lands near or above 0.92
#       -> the Phase 1 gap was the recipe, not the architecture. Do NOT run the
#          remaining ~60h expecting a different answer. The honest thesis
#          becomes "cohort pre-training closes the single-channel gap, realised
#          at 1.3-4.3x lower compute", with the architecture demoted to a
#          cost argument -- which the MAC numbers already support on their own
#          (585,920 vs 2,523,328).
#
# Either outcome is publishable. Only one of them is the paper currently being
# written, which is why this runs before the other sixty hours.
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

MODEL=baseline_frontiers2d
SEED=0
LOG="${WEARSEIZURE_ARTIFACTS_DIR}/phase2_probe.log"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 2 probe start. commit=$(git rev-parse --short HEAD) model=$MODEL seed=$SEED"

# IMPORTANT: this trains into a directory the un-pre-trained frontiers2d run
# already occupies (seed0). Rows 2/4/6/8 came from those checkpoints, so they
# are moved aside rather than overwritten -- the -L1 arm of the ablation grid
# still needs them, and they cost GPU hours to make.
BASE="$WEARSEIZURE_ARTIFACTS_DIR/$MODEL/patient_specific_loso_edf/w4s_stride1s"
if [ -d "$BASE/seed0" ] && [ ! -d "$BASE/seed0_noL1" ]; then
  say "preserving the existing no-L1 frontiers2d run as seed0_noL1"
  mv "$BASE/seed0" "$BASE/seed0_noL1"
fi
mkdir -p "$BASE/seed0"

say "cohort pre-training inits for $MODEL seed=$SEED (~6.4h, 3 shards)"
for i in 0 1 2; do
  python scripts/pretrain_cohort.py profile=server data=chbmit \
    model="$MODEL" seed="$SEED" +shard="$i" +n_shards=3 profile.num_workers=4 \
    >> "$LOG" 2>&1 &
done
wait
n_init=$(ls "$WEARSEIZURE_ARTIFACTS_DIR/pretrain/$MODEL/w4s_stride1s/seed$SEED"/*.pt 2>/dev/null | wc -l)
say "inits: $n_init/13"
[ "$n_init" -eq 13 ] || { say "ABORT: expected 13 inits, found $n_init"; exit 1; }

say "training 66 folds for $MODEL seed=$SEED with L1 (~3.8h)"
run python scripts/train.py profile=server data=chbmit \
  model="$MODEL" seed="$SEED" train.pretrain.enabled=true
n_folds=$(ls "$BASE/seed0"/*.metrics.json 2>/dev/null | wc -l)
say "folds: $n_folds/66"
[ "$n_folds" -eq 66 ] || { say "ABORT: expected 66 folds, found $n_folds"; exit 1; }

# Same post-processing as the Phase 1 numbers it will be compared against, so
# the only thing that differs between the two is the architecture.
say "scoring with the row-24 post-processing config"
run python scripts/rethreshold.py profile=server data=chbmit \
  model="$MODEL" seed="$SEED" postprocess=hysteresis_widegrid \
  postprocess.run_length=2 postprocess.ema_alpha=0.25
run python scripts/evaluate.py profile=server data=chbmit \
  model="$MODEL" postprocess=hysteresis_widegrid \
  postprocess.run_length=2 postprocess.ema_alpha=0.25 \
  profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml

# One seed against three is not a significance test, and the script says so
# rather than pretending otherwise. It is a magnitude check: is frontiers2d+L1
# sitting near 0.88 or near 0.92?
say "single-seed comparison vs k5only+L1 (3 seeds) -- MAGNITUDE ONLY, not a significance test"
ART="$WEARSEIZURE_ARTIFACTS_DIR"
run python scripts/paired_bootstrap.py \
  "$ART/wearseizure1d_k5only/patient_specific_loso_edf/w4s_stride1s" \
  "$BASE/seed0" \
  --all-metrics --json "$ART/probe_k5onlyL1_vs_frontiers2dL1.json"

say "Probe done. Send back $LOG and probe_k5onlyL1_vs_frontiers2dL1.json"
