#!/usr/bin/env bash
# Measure what `train.compile_mode=reduce-overhead` is actually worth.  (~30min)
#
# Why this has to be measured, not assumed
# ----------------------------------------
# docs/SERVER_INVENTORY.md establishes that the binding constraint on this
# workload is CUDA kernel-launch overhead: a batch of 256 through a ~12k-
# parameter model is ~60us of arithmetic issued as roughly a hundred tiny
# kernels, so the device spends its time between kernels rather than inside
# them. That is why GPU-utilisation and memory-bandwidth counters both read
# 20-30% however many DataLoader workers run, and why adding workers cannot fix
# it.
#
# `reduce-overhead` captures the step into CUDA graphs and replays it, attacking
# that directly. The inventory doc predicts it "should beat sharding" -- but
# records it as off by default "until measured against an uncompiled run",
# because nothing feeding a publication should run on a predicted speedup.
#
# Measuring it needs an idle machine, which is a rare condition on a shared box.
# Hence this script.
#
# What it does NOT measure: whether compiled and uncompiled produce identical
# metrics. They should -- CUDA graphs replay the same kernels, and
# training/loop.unwrap_compiled keeps the `_orig_mod.` prefix out of the
# checkpoint -- but the script prints both runs' per-fold sensitivity so a
# divergence would be visible rather than silent.
#
# Throwaway seeds 98/99 are used so nothing that feeds a result is touched, and
# pre-training is OFF: the question is about the per-epoch train loop, which is
# the same either way, and building 13 cohort inits for a throwaway seed would
# cost 6 hours to measure nothing.
#
#   bash scripts/measure_compile_speedup.sh          # then delete seed98/seed99
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

MODEL=wearseizure1d_k5only
FOLDS=3
LOG="$WEARSEIZURE_ARTIFACTS_DIR/compile_speedup.log"

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# Run the machine idle for this. A concurrent job turns the measurement into a
# measurement of contention.
say "GPU state before starting -- anything else running invalidates this"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$LOG"

timed() {                        # timed <label> <seed> [extra overrides...]
  local label="$1" seed="$2"; shift 2
  say "$label: $MODEL, $FOLDS folds, seed=$seed"
  local t0 t1
  t0=$(date +%s)
  python scripts/train.py profile=server data=chbmit \
    model="$MODEL" seed="$seed" train.pretrain.enabled=false \
    train.max_folds="$FOLDS" train.force_retrain=true "$@" >> "$LOG" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ "$rc" -ne 0 ]; then
    say "$label FAILED (exit $rc) -- see $LOG. torch.compile can fall back or error on small graphs."
    return 1
  fi
  say "$label: $((t1 - t0))s for $FOLDS folds  ->  $(( (t1 - t0) / FOLDS ))s per fold"
  echo "$((t1 - t0))"
}

t_plain=$(timed "UNCOMPILED baseline" 99 | tail -1)
t_compiled=$(timed "COMPILED reduce-overhead" 98 train.compile_mode=reduce-overhead | tail -1)

say "Result"
# Guard on "is a number", not just "is non-empty": a torch.compile failure makes
# timed() return its error message, and comparing that with -gt is a shell error
# rather than a clean "no result".
is_num() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
if is_num "${t_plain:-}" && is_num "${t_compiled:-}" && [ "$t_compiled" -gt 0 ]; then
  # Integer arithmetic to two decimals without bc, which is not always present.
  speedup=$(( t_plain * 100 / t_compiled ))
  printf 'uncompiled %ss | compiled %ss | speedup %d.%02dx\n' \
    "$t_plain" "$t_compiled" $((speedup / 100)) $((speedup % 100)) | tee -a "$LOG"
  cat <<'EOF' | tee -a "$LOG"

How to act on it:
  >= 1.5x  adopt it for Phase 2 stage B; it compounds over ~400 fold-trainings
  1.1-1.5x adopt only if the per-fold metrics below match the uncompiled run
  <= 1.1x  leave it off; the win is not worth a numerics risk on paper results
EOF
else
  say "One of the runs did not complete; no speedup figure. Read $LOG."
fi

say "Per-fold sensitivity, both runs -- these should agree"
for seed in 99 98; do
  d="$WEARSEIZURE_ARTIFACTS_DIR/$MODEL/patient_specific_loso_edf/w4s_stride1s/seed$seed"
  echo "seed$seed:" | tee -a "$LOG"
  python - "$d" <<'PYEOF' | tee -a "$LOG"
import json, sys, glob, os
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.metrics.json"))):
    m = json.load(open(f))["test_event_metrics"]
    print(f"  {os.path.basename(f):40s} sens={m['sensitivity']:.4f} FAR/h={m['far_per_hour']:.4f}")
PYEOF
done

say "Done. Throwaway artifacts to remove once you have read the numbers:"
echo "  rm -rf '$WEARSEIZURE_ARTIFACTS_DIR/$MODEL/patient_specific_loso_edf/w4s_stride1s/seed98'" | tee -a "$LOG"
echo "  rm -rf '$WEARSEIZURE_ARTIFACTS_DIR/$MODEL/patient_specific_loso_edf/w4s_stride1s/seed99'" | tee -a "$LOG"
