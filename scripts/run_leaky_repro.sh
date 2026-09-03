#!/usr/bin/env bash
# Item A7 -- the leaky-protocol reproduction, shardable across servers.
#
# What it proves
# --------------
# Two things, which are the two the project needs:
#
#   1. Our reproductions of the published architectures reach the published
#      NUMBERS when run under the published PROTOCOL. Without this, the claim
#      that the gap is protocol-caused is an inference and a reviewer can answer
#      "or your reproduction is simply worse".
#   2. Our own model, run under that same protocol, behaves the same way -- so
#      the effect is a property of the evaluation, not of any one architecture.
#
# It also decomposes the gap instead of asserting it. Four factors separate the
# published figure from ours, and only the first is dishonest:
#
#   A  random-window split + global norm + threshold on test + per-window metric
#   B  recording split     + global norm + threshold on test + per-window metric
#   C  recording split     + train-only  + threshold on val  + per-window metric
#   D  recording split     + train-only  + threshold on val  + per-EVENT metric
#
# A->B is the leak, B->C is the fitting leak, C->D is what is being counted.
# Rung D is this project's own protocol and is NOT run here: its numbers are
# already in docs/EXPERIMENT_LOG_G1a.md.
#
# Sharding
# --------
# SERVER-03 and SERVER-04 share ~/Manh over NFS, so both write to the same
# artifacts tree. Every run writes to leaky_repro/<rung>/<model>/, so shards
# that differ in rung or model can never collide. Pick a shard per host:
#
#   SERVER-03:  SHARD=repro  bash scripts/run_leaky_repro.sh
#   SERVER-04:  SHARD=ours   bash scripts/run_leaky_repro.sh
#
#   repro   the two reproduced baselines at rung A, plus rungs B and C on
#           frontiers2d -- that is, the published comparison and its decomposition
#   ours    our own model across all three rungs
#
#   SHARD=all runs everything on one host, for when only one is free.
#
#   SHARD=repro nohup bash scripts/run_leaky_repro.sh > "leaky_$(hostname).out" 2>&1 &
#
# The output file MUST carry the hostname. ~/Manh is NFS: the checkout itself
# is one shared directory, so plain `> leaky_repro.out` from two hosts has
# each truncating the other's file -- which is how SERVER-04's startup error
# was lost and its log showed SERVER-03's output instead. The script's own
# tee log is already host-qualified; this is the nohup redirect.
set -uo pipefail

# These two are what configs/profile/server.yaml interpolates. They live in
# SERVER-02's ~/.bashrc, so moving to a fresh host is exactly where they go
# missing -- and a terse "set it first" sends someone hunting through configs.
missing=""
[ -z "${CHBMIT_RAW_DIR:-}" ] && missing="$missing CHBMIT_RAW_DIR"
[ -z "${WEARSEIZURE_ARTIFACTS_DIR:-}" ] && missing="$missing WEARSEIZURE_ARTIFACTS_DIR"
if [ -n "$missing" ]; then
  echo "ABORT: not set:$missing"
  echo
  echo "configs/profile/server.yaml interpolates both. On the shared NFS layout:"
  echo "  export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0"
  echo "  export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts"
  echo
  echo "Note ART is not one of them -- exporting ART alone leaves both unset."
  exit 1
fi
[ -d "$CHBMIT_RAW_DIR" ] || { echo "ABORT: CHBMIT_RAW_DIR does not exist: $CHBMIT_RAW_DIR"; exit 1; }
[ -d "$WEARSEIZURE_ARTIFACTS_DIR" ] || { echo "ABORT: WEARSEIZURE_ARTIFACTS_DIR does not exist: $WEARSEIZURE_ARTIFACTS_DIR"; exit 1; }

ulimit -n 65536 2>/dev/null
if [ "$(ulimit -n)" -lt 65536 ]; then
  echo "ABORT: could not raise the open-file limit past $(ulimit -n)."
  echo "This is what killed the first L4 attempt at the fourth cohort init of every"
  echo "combination. Raise the hard limit, or put 'ulimit -n 65536' in ~/.bashrc."
  echo "Current hard limit: $(ulimit -Hn)"
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Hydra names its run directory by timestamp-to-the-second, and these hosts
# share the artifacts tree, so the host has to be part of the name.
export WEARSEIZURE_RUN_HOST="$(hostname)"

ART="$WEARSEIZURE_ARTIFACTS_DIR"
SHARD="${SHARD:-all}"
LOG="$ART/leaky_repro_${SHARD}_$(hostname).log"
WORKERS="${WORKERS:-}"
WORKER_ARG=()
[ -n "$WORKERS" ] && WORKER_ARG=(profile.num_workers="$WORKERS")

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

# (rung, model) pairs. Rung A on all three models answers the two questions;
# rungs B and C on frontiers2d alone supply the decomposition, because
# frontiers2d is the architecture whose published number is being reproduced.
case "$SHARD" in
  repro) JOBS=(
    "A_as_published baseline_frontiers2d"
    "A_as_published baseline_compact1d_7k"
    "B_split_by_recording baseline_frontiers2d"
    "C_no_fitting_leak baseline_frontiers2d"
  ) ;;
  ours)  JOBS=(
    "A_as_published wearseizure1d_k5only"
    "B_split_by_recording wearseizure1d_k5only"
    "C_no_fitting_leak wearseizure1d_k5only"
  ) ;;
  all)   JOBS=(
    "A_as_published baseline_frontiers2d"
    "A_as_published baseline_compact1d_7k"
    "A_as_published wearseizure1d_k5only"
    "B_split_by_recording baseline_frontiers2d"
    "C_no_fitting_leak baseline_frontiers2d"
    "B_split_by_recording wearseizure1d_k5only"
    "C_no_fitting_leak wearseizure1d_k5only"
  ) ;;
  *) echo "SHARD must be repro, ours or all; got '$SHARD'"; exit 1 ;;
esac

say "A7 leaky reproduction. shard=$SHARD host=$(hostname) commit=$(git rev-parse --short HEAD) workers=${WORKERS:-<profile default>}"

for job in "${JOBS[@]}"; do
  set -- $job
  rung="$1"; model="$2"
  say "rung=$rung model=$model"
  run python scripts/run_leaky_repro.py profile=server data=chbmit \
    model="$model" "+rung=$rung" "${WORKER_ARG[@]}"

  n=$(ls "$ART/leaky_repro/$rung/$model"/*.json 2>/dev/null | wc -l)
  say "folds $rung/$model: $n/66"
  # Same rule as every other phase: a partial cohort is not a result. The first
  # L4 attempt reported sensitivity 1.0000 from 13 folds over 3 easy patients.
  [ "$n" -eq 66 ] || { say "ABORT: $rung/$model has $n/66 folds"; exit 1; }
done

say "shard $SHARD done. Summarise once BOTH shards have finished:"
say "  python scripts/summarise_leaky_repro.py \$WEARSEIZURE_ARTIFACTS_DIR --markdown"
