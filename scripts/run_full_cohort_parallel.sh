#!/usr/bin/env bash
# The 24-case cohort, saturating one server instead of spreading over four.
#
#   nohup bash scripts/run_full_cohort_parallel.sh > full_cohort.out 2>&1 &
#
# Where the speed comes from
# --------------------------
# Unlike the training-heavy phases, this run is dominated by DATA PREPARATION,
# not by the GPU. A fold costs about 240 s, of which roughly 180 s is reading
# four channels of EDF and filtering them -- work that happens in the main
# process, single-threaded, and never reaches a DataLoader worker.
#
# That changes the arithmetic. Concurrent processes parallelise the part that
# actually dominates, so throughput rises close to linearly with POOL until one
# of three ceilings is hit:
#
#   * 14 physical cores. Past POOL=14 the data-prep phases contend directly.
#   * NFS bandwidth. All servers read ~/Manh over one mount, and this run reads
#     four channels per fold rather than one. This is the ceiling most likely to
#     bind, and the only way to know is to watch s/fold as POOL rises.
#   * Memory. ~3 GB resident per process here (four channels of one patient,
#     raw plus filtered) against 188-251 GB. Not a constraint at any usable POOL.
#
# DataLoader workers are divided BETWEEN processes, never multiplied by them:
# workers slice arrays and build tensors, which is memory-bandwidth bound, so
# hyperthread siblings contend for the same load/store units.
#
# Knobs
# -----
#   POOL=6        concurrent shards (default 6)
#   SEEDS='[0]'   seeds to run
#   SHARD_BASE=0  set on a second host so its shards do not repeat this one's
#   TOTAL=        total shard count across all hosts (defaults to POOL)
#
# Two hosts, 6 shards each:
#   host A:  POOL=6 TOTAL=12 SHARD_BASE=0  bash scripts/run_full_cohort_parallel.sh
#   host B:  POOL=6 TOTAL=12 SHARD_BASE=6  bash scripts/run_full_cohort_parallel.sh
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CORES=14
POOL="${POOL:-6}"
TOTAL="${TOTAL:-$POOL}"
SHARD_BASE="${SHARD_BASE:-0}"
SEEDS="${SEEDS:-[0]}"
MODEL="${MODEL:-wearseizure1d_k5only}"

WORKERS=$(( CORES / POOL )); [ "$WORKERS" -lt 1 ] && WORKERS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
HOST=$(hostname -s)
LOGDIR="$ART/full_cohort_logs"
mkdir -p "$LOGDIR"

if [ $(( SHARD_BASE + POOL )) -gt "$TOTAL" ]; then
  echo "SHARD_BASE=$SHARD_BASE + POOL=$POOL exceeds TOTAL=$TOTAL." >&2
  echo "Shards would repeat across hosts, and two processes computing one fold" >&2
  echo "race to write the same file. Fix TOTAL or SHARD_BASE." >&2
  exit 1
fi

echo "=== [$(date '+%F %T')] $HOST: full cohort, commit=$(git rev-parse --short HEAD)"
echo "    POOL=$POOL of TOTAL=$TOTAL shards (base $SHARD_BASE), ${WORKERS} workers each"
echo "    seeds=$SEEDS model=$MODEL"
echo "    logs: $LOGDIR/${HOST}_shard*.log"

pids=()
for (( k = 0; k < POOL; k++ )); do
  idx=$(( SHARD_BASE + k ))
  log="$LOGDIR/${HOST}_shard${idx}.log"
  # Host-qualified hydra run dir: the four servers share ~/Manh over NFS and the
  # timestamp resolves to the second, so two hosts starting in the same second
  # would write their .hydra/ into one directory.
  python scripts/run_full_cohort.py \
      profile=server data=chbmit model="$MODEL" \
      "train.seeds=$SEEDS" \
      +shard.index="$idx" +shard.count="$TOTAL" \
      profile.num_workers="$WORKERS" \
      > "$log" 2>&1 &
  pids+=($!)
  echo "    shard $idx -> pid ${pids[-1]}"
  # Stagger the starts: every shard opens by reading EDF, and launching six
  # simultaneous readers is the one moment this run can saturate NFS on its own.
  sleep 10
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "!! pid $pid exited non-zero"; fail=1; }
done

echo "=== [$(date '+%F %T')] $HOST: done, $(ls "$ART"/full_cohort/seed*/*.json 2>/dev/null | wc -l) folds on disk"
echo "    python scripts/summarise_full_cohort.py $ART"
exit "$fail"
