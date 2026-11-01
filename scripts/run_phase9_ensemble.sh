#!/usr/bin/env bash
# Phase 9 -- inference-time seed ensemble of the headline configuration (L8).
#
# The question
# ------------
# L8's three seeds give fold-averaged sensitivity 0.9848, 0.9242 and 0.9394:
# one to five of the 66 folds miss their seizure depending on the draw. Does
# averaging the three seeds' logits at inference buy a better detector than
# deploying any one seed? No training: every member checkpoint already exists.
#
# What decides it
# ---------------
# The paired bootstrap of the ensemble against the THREE-SEED MEAN of L8 --
# i.e. against what deploying one seed delivers on average -- clustered by
# patient. Not against the best seed: choosing seed 0 after seeing its test
# result is the selection the paper criticises.
#
# The ensemble costs 3x the MACs and the weight memory of one model. That is
# affordable on the KV260 (about 0.6 ms per one-second hop), but it is a cost,
# and an interval spanning zero means it buys nothing for it.
#
#   nohup bash scripts/run_phase9_ensemble.sh > phase9.out 2>&1 &
#   tail -f phase9.out
set -uo pipefail

ENV_FILE="$(git rev-parse --show-toplevel 2>/dev/null || echo .)/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

: "${WEARSEIZURE_ARTIFACTS_DIR:?not set, and no .env at the repo root defines it}"
: "${CHBMIT_RAW_DIR:?not set, and no .env at the repo root defines it}"

if ! ulimit -n 65536 2>/dev/null && [ "$(ulimit -n)" -lt 8192 ]; then
  echo "ABORT: open-file limit is $(ulimit -n); raise it or put 'ulimit -n 65536' in ~/.bashrc."
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase9_ensemble.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
MODEL=wearseizure1d_k5only
SOURCE_TAG="${SOURCE_TAG:-L8}"
SEEDS=(0 1 2)
ENS_TAG="${SOURCE_TAG}ens$(printf '%s' "${SEEDS[@]}")"
SRC_ROOT="$ART/$MODEL/$SPLIT/${WINDOW}__${SOURCE_TAG}"
ENS_ROOT="$ART/$MODEL/$SPLIT/${WINDOW}__${ENS_TAG}"
SEEDS_HYDRA="[$(IFS=,; echo "${SEEDS[*]}")]"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 9 start. commit=$(git rev-parse --short HEAD) source=$SOURCE_TAG seeds=${SEEDS[*]} -> $ENS_TAG"

# Every member must be complete before anything is scored. ensemble_seeds.py
# refuses a partial ensemble too, but finding out after loading the corpus
# wastes the load.
for seed in "${SEEDS[@]}"; do
  p=$(ls "$SRC_ROOT/seed$seed"/*.pt 2>/dev/null | wc -l)
  m=$(ls "$SRC_ROOT/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
  [ "$p" -eq 66 ] && [ "$m" -eq 66 ] || { say "ABORT: $SOURCE_TAG seed=$seed has $p/66 checkpoints, $m/66 metrics"; exit 1; }
done
say "preconditions met: $SOURCE_TAG 66/66 checkpoints and metrics for seeds ${SEEDS[*]}"

say "score: ensemble of $MODEL $SOURCE_TAG seeds ${SEEDS[*]}, thresholds re-searched on validation"
run python scripts/ensemble_seeds.py profile=server data=chbmit model="$MODEL" \
  "train.seeds=$SEEDS_HYDRA" train.run_tag="$SOURCE_TAG" postprocess=hysteresis_widegrid

n=$(ls "$ENS_ROOT/seed0"/*.metrics.json 2>/dev/null | wc -l)
say "ensemble folds: $n/66"
# A partial cohort once reported sensitivity 1.0000 -- three easy patients.
[ "$n" -eq 66 ] || { say "ABORT: ensemble has $n/66 folds; do not read any summary until it is 66"; exit 1; }

say "evaluate: ensemble $ENS_TAG"
run python scripts/evaluate.py profile=server data=chbmit model="$MODEL" \
  train.run_tag="$ENS_TAG" seed=0 postprocess=hysteresis_widegrid \
  profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml

say "paired bootstrap: ensemble vs the $SOURCE_TAG three-seed mean (the decision)"
run python scripts/paired_bootstrap.py "$ENS_ROOT" "$SRC_ROOT" \
  --all-metrics --json "$ART/paired_${MODEL}_${ENS_TAG}_vs_${SOURCE_TAG}.json"

say "Phase 9 done. Send back $LOG and $ART/paired_${MODEL}_${ENS_TAG}_vs_${SOURCE_TAG}.json"
