#!/usr/bin/env bash
# Phase 2 stage B, parallel: three L1 seeds for every architecture.
#
# Replaces the sequential stage B inside run_phase2_full.sh, for the case where
# SERVER-02 is not shared. Same work, same results, roughly a third of the
# wall-clock.
#
# Where the speed comes from -- and where it does not
# --------------------------------------------------
# The binding constraint here is CUDA kernel-launch overhead, not arithmetic and
# not data loading (docs/SERVER_INVENTORY.md, measured: ~60us of arithmetic per
# batch of 256, issued as ~100 tiny kernels; the loader at 14 workers is already
# 5x faster than needed). So:
#
#   - Concurrent PROCESSES help, because independent CUDA streams interleave
#     each other's launch gaps. This is the lever the script pulls.
#   - More DataLoader workers do NOT help, and past 14 they hurt: workers slice
#     arrays and build tensors, which is memory-bandwidth bound, so hyperthread
#     siblings contend for the same load/store units. Workers are therefore
#     divided BETWEEN concurrent processes, never multiplied by them.
#   - A bigger batch would halve launches per epoch, but it changes optimisation
#     semantics and needs the LR retuned. Not a free throughput knob, not used.
#
# Memory is not the limit: ~2.2GB resident per process after the float32 change,
# on 188GB, and ~400MB of VRAM per CUDA context on 48GB.
#
# Knobs
# -----
#   POOL=6           concurrent pre-training processes (default 6)
#   TRAIN_POOL=3     concurrent training processes (default 3)
#   COMPILE_MODE=    e.g. reduce-overhead, if measure_compile_speedup.sh earned it
#
# Workers per process are derived from the pool so the total stays at or under
# the 14 physical cores. Pre-training gets the larger pool because each of its
# processes is short and independent; training gets fewer, larger processes.
#
#   nohup bash scripts/run_phase2b_parallel.sh > phase2b.out 2>&1 &
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase2b_parallel.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
CORES=14

POOL="${POOL:-6}"
TRAIN_POOL="${TRAIN_POOL:-3}"
COMPILE_MODE="${COMPILE_MODE:-}"

PRE_WORKERS=$(( CORES / POOL ));   [ "$PRE_WORKERS"   -lt 1 ] && PRE_WORKERS=1
TRAIN_WORKERS=$(( CORES / TRAIN_POOL )); [ "$TRAIN_WORKERS" -lt 1 ] && TRAIN_WORKERS=1

# (model, seed) combinations still missing an L1 run. k5only already has 0,1,2
# from Phase 1; frontiers2d has seed 0 from the probe.
COMBOS=(
  "baseline_frontiers2d 1"
  "baseline_frontiers2d 2"
  "baseline_compact1d_7k 0"
  "baseline_compact1d_7k 1"
  "baseline_compact1d_7k 2"
)
ALL_MODELS=(wearseizure1d_k5only baseline_frontiers2d baseline_compact1d_7k)

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

# Bounded-concurrency pool. `wait -n` returns as soon as ANY child finishes, so
# a slot is refilled immediately instead of waiting for a whole batch -- which
# matters because the per-combo work is uneven (frontiers2d is 2.5M MACs against
# compact1d_7k's 1.4M).
pool_run() {                      # pool_run <max> -- reads commands on stdin
  local max="$1" running=0 line
  while IFS= read -r line; do
    eval "$line" &
    running=$(( running + 1 ))
    if [ "$running" -ge "$max" ]; then wait -n; running=$(( running - 1 )); fi
  done
  wait
}

say "Phase 2b start. commit=$(git rev-parse --short HEAD)"
say "pools: pretrain ${POOL}x${PRE_WORKERS} workers, train ${TRAIN_POOL}x${TRAIN_WORKERS} workers, compile='${COMPILE_MODE:-off}'"

# ---------------------------------------------------------------------------
# Preserve the un-pre-trained seed 0 runs before anything writes over them.
# compact1d_7k seed 0 is the source of rows 1/3/5/7/13; those checkpoints are
# the -L1 arm of the ablation grid and cost GPU hours to make.
# ---------------------------------------------------------------------------
for combo in "${COMBOS[@]}"; do
  set -- $combo
  d="$ART/$1/$SPLIT/$WINDOW"
  if [ "$2" = "0" ] && [ -d "$d/seed0" ] && [ ! -d "$d/seed0_noL1" ]; then
    say "preserving existing no-L1 $1 run as seed0_noL1"
    mv "$d/seed0" "$d/seed0_noL1"
    mkdir -p "$d/seed0"
  fi
done

# ---------------------------------------------------------------------------
# Stage 1 -- all cohort pre-training, as one flat pool of (combo, shard) jobs.
#
# Flattening across combos rather than finishing one combo at a time is the
# whole point: 5 combos x 13 subjects is 65 independent pre-trainings, each
# writing its own cache file, so there is nothing to serialise and no lock
# needed. Sequentially this is ~32h; at POOL=6 it is a single pass.
# ---------------------------------------------------------------------------
N_SHARDS="$POOL"
say "Stage 1: pre-training inits, ${#COMBOS[@]} combos x $N_SHARDS shards, pool=$POOL"
{
  for combo in "${COMBOS[@]}"; do
    set -- $combo
    for i in $(seq 0 $(( N_SHARDS - 1 ))); do
      echo "python scripts/pretrain_cohort.py profile=server data=chbmit model=$1 seed=$2 +shard=$i +n_shards=$N_SHARDS profile.num_workers=$PRE_WORKERS >> '$LOG' 2>&1"
    done
  done
} | pool_run "$POOL"

for combo in "${COMBOS[@]}"; do
  set -- $combo
  n=$(ls "$ART/pretrain/$1/$WINDOW/seed$2"/*.pt 2>/dev/null | wc -l)
  say "inits $1 seed=$2: $n/13"
  [ "$n" -eq 13 ] || { say "ABORT: $1 seed=$2 has $n/13 inits"; exit 1; }
done

# ---------------------------------------------------------------------------
# Stage 2 -- training, one process per combo, fewer and larger than stage 1.
# ---------------------------------------------------------------------------
say "Stage 2: training ${#COMBOS[@]} combos, pool=$TRAIN_POOL"
COMPILE_ARG=""
[ -n "$COMPILE_MODE" ] && COMPILE_ARG="train.compile_mode=$COMPILE_MODE"
{
  for combo in "${COMBOS[@]}"; do
    set -- $combo
    echo "python scripts/train.py profile=server data=chbmit model=$1 seed=$2 train.pretrain.enabled=true profile.num_workers=$TRAIN_WORKERS $COMPILE_ARG >> '$LOG' 2>&1"
  done
} | pool_run "$TRAIN_POOL"

for combo in "${COMBOS[@]}"; do
  set -- $combo
  n=$(ls "$ART/$1/$SPLIT/$WINDOW/seed$2"/*.metrics.json 2>/dev/null | wc -l)
  say "folds $1 seed=$2: $n/66"
  [ "$n" -eq 66 ] || { say "ABORT: $1 seed=$2 has $n/66 folds"; exit 1; }
done

# ---------------------------------------------------------------------------
# Stage 3 -- score all three architectures on one shared configuration, so the
# only thing differing between them is the architecture. Re-thresholding is
# forced to num_workers=0 inside engine_baseline, so this stage is CPU-light and
# parallelises freely.
# ---------------------------------------------------------------------------
say "Stage 3: rethreshold all architectures x 3 seeds on the row 22 config"
{
  for model in "${ALL_MODELS[@]}"; do
    for seed in 0 1 2; do
      echo "python scripts/rethreshold.py profile=server data=chbmit model=$model seed=$seed postprocess=hysteresis_widegrid >> '$LOG' 2>&1"
    done
  done
} | pool_run "$POOL"

for model in "${ALL_MODELS[@]}"; do
  say "Stage 3 evaluate: $model, 3 seeds, v2 gates"
  run python scripts/evaluate.py profile=server data=chbmit \
    model="$model" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
    profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml
done

K5="$ART/wearseizure1d_k5only/$SPLIT/$WINDOW"
for other in baseline_frontiers2d baseline_compact1d_7k; do
  say "paired bootstrap: k5only vs $other, 3 seeds each"
  run python scripts/paired_bootstrap.py "$K5" "$ART/$other/$SPLIT/$WINDOW" \
    --all-metrics --json "$ART/paired_k5only_vs_${other}_3seed.json"
done

say "Phase 2b done. Send back $LOG and the paired_*_3seed.json files"
