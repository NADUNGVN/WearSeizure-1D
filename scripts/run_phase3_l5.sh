#!/usr/bin/env bash
# Phase 3 -- lever L5: widen the cohort pre-training corpus to the rest of CHB-MIT.
#
# Why this and not something else
# -------------------------------
# Phase 2 settled what actually moves this project. Architecture does not: the
# three architectures are statistically indistinguishable on macro sensitivity,
# FAR and delay, and the false-alarm advantage k5only appeared to have turned
# out to be one patient (chb23). Cohort pre-training does: it lifted
# frontiers2d by 8.9pp, 0.8811 -> 0.9726, with a seed std of 0.0037.
#
# L5 is more of exactly that. The 13-case restriction exists because Chung et
# al. clinically confirmed seizure onset was observable from one wearable
# position in those cases -- a requirement for SCORING a single-channel
# detector, not for teaching it what ictal EEG looks like. So evaluation stays
# at 13 cases / 66 folds / 185.0h while pre-training draws on the rest.
#
# What is still missing, and what L5 is aimed at:
#   M2 sensitivity >= 0.95   k5only 0.9358   short 1.4pp
#   M3 FAR         <= 0.20   k5only 0.2261   short 0.026   -- no architecture passes
#   M5 worst-pt FAR<= 0.50   k5only 0.7785   short 0.28    -- no architecture passes
#   M7 delay       <= 17s    k5only 18.83    short 1.8s
#
# Run on BOTH k5only and frontiers2d, three seeds each, so the question "does
# more pre-training data help every architecture or only the cheap one" gets an
# answer rather than an assumption.
#
# Writes to `<window>__L5/`, NOT over rows 32-34. Those are the control arm and
# the whole comparison depends on them surviving.
#
#   nohup bash scripts/run_phase3_l5.sh > phase3.out 2>&1 &
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase3_l5.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s
TAG=L5
MODELS=(wearseizure1d_k5only baseline_frontiers2d)
SEEDS=(0 1 2)

# Concurrency is lower here than in Phase 2 because each process holds the whole
# pre-training corpus resident and L5 makes that corpus 3.7x larger. The first
# attempt ran POOL=3 and was OOM-killed after 10 hours with 35 of 78 inits
# built: measured 72.6 GiB per process, against a check that had predicted 7.2.
# The allocations behind that gap are fixed; POOL=2 is the margin for the part
# of the estimate that is still an estimate.
POOL="${POOL:-2}"
CORES=14
WORKERS=$(( CORES / POOL )); [ "$WORKERS" -lt 1 ] && WORKERS=1

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
pool_run() {
  local max="$1" running=0 line
  while IFS= read -r line; do
    eval "$line" &
    running=$(( running + 1 ))
    if [ "$running" -ge "$max" ]; then wait -n; running=$(( running - 1 )); fi
  done
  wait
}

say "Phase 3 (L5) start. commit=$(git rev-parse --short HEAD) pool=${POOL}x${WORKERS} workers"

# ---------------------------------------------------------------------------
# Step 1 -- build the wider pre-training manifest, and refuse to continue if it
# does not fit in memory.
#
# This reads and hashes every EDF, so it is slow (tens of minutes) but it is
# read-only with respect to the clinical data. `data.pretrain_channels` defaults
# to all four wearable positions per case, which multiplies the resident signal
# by roughly four -- that is a deliberate choice (no position is clinically
# confirmed for these cases, and a position-agnostic representation matches a
# wearable whose electrode sits at one of four spots) but it is also the reason
# the memory figure has to be checked rather than assumed.
# ---------------------------------------------------------------------------
say "Step 1: building manifests (this also rebuilds the 13-case evaluation manifest, identically)"
run python scripts/make_manifest.py profile=server data=chbmit

PRETRAIN_CSV="$ART/manifest/chbmit_pretrain_manifest.csv"
[ -f "$PRETRAIN_CSV" ] || { say "ABORT: $PRETRAIN_CSV was not produced"; exit 1; }

say "Step 1b: corpus size and memory check"
AVAIL_GIB=$(free -g | awk '/^Mem:/ {print $7}')
python - "$PRETRAIN_CSV" "$ART/manifest/chbmit_manifest.csv" "$POOL" "$AVAIL_GIB" <<'PYEOF' | tee -a "$LOG"
import sys, pandas as pd
pre = pd.read_csv(sys.argv[1], keep_default_na=False, na_values=[])
ev  = pd.read_csv(sys.argv[2], keep_default_na=False, na_values=[])
pool, avail = int(sys.argv[3]), float(sys.argv[4])

# EMPIRICAL, not derived. The first version of this check computed 3.52 MiB per
# corpus-hour from "float32 at 256 Hz" and cleared a configuration that the OOM
# killer then destroyed: measured anon-rss was 72.6 GiB per process against a
# 2085h corpus, i.e. 35.6 MiB/h -- ten times the theoretical figure, because
# build_fold_datasets held four full copies and fit_affine_normalizer allocated
# two more float64 temporaries the size of the train partition.
#
# Those allocations are gone (bit-identically), which removes roughly half the
# peak, so 18 MiB/h is the post-fix figure. It is still a measurement scaled by
# an estimate, so the margin below is deliberately generous -- a wrong guess
# here costs ten hours of GPU time and an OOM, not a warning.
MIB_PER_CORPUS_HOUR = 18.0
h_pre, h_ev = pre["duration_sec"].sum() / 3600, ev["duration_sec"].sum() / 3600
per_proc = (h_pre + h_ev) * MIB_PER_CORPUS_HOUR / 1024
budget = 0.6 * avail
print(f"  evaluation corpus  : {len(ev):5d} rows, {ev['subject_id'].nunique():2d} cases, {h_ev:8.1f}h")
print(f"  L5 pre-train corpus: {len(pre):5d} rows, {pre['subject_id'].nunique():2d} cases, {h_pre:8.1f}h")
print(f"  cases              : {sorted(pre['subject_id'].unique())}")
print(f"  positions          : {sorted(pre['channel_name'].unique())}")
print(f"  estimated resident : {per_proc:.1f} GiB/process x{pool} = {per_proc * pool:.1f} GiB")
print(f"  RAM available now  : {avail:.0f} GiB, budget at 60% = {budget:.0f} GiB")
sys.exit(0 if per_proc * pool < budget else 3)
PYEOF
rc=$?
if [ "$rc" -eq 3 ]; then
  say "ABORT: the corpus does not fit at POOL=$POOL. Re-run with a smaller POOL, or"
  say "       narrow the corpus, e.g. data.pretrain_channels=[P8-O2] (see configs/data/chbmit.yaml)."
  exit 1
elif [ "$rc" -ne 0 ]; then
  say "ABORT: the corpus check itself failed (exit $rc)"; exit 1
fi

L5="train.pretrain.enabled=true train.pretrain.use_wider_corpus=true train.run_tag=$TAG"

# ---------------------------------------------------------------------------
# Step 2 -- pre-training. Cached inits carry the hash of the corpus that made
# them, so the Phase 2 inits are correctly rejected rather than silently reused;
# every one of these is trained fresh.
# ---------------------------------------------------------------------------
say "Step 2: cohort pre-training on the wider corpus"
{
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for i in $(seq 0 $(( POOL - 1 ))); do
        echo "python scripts/pretrain_cohort.py profile=server data=chbmit model=$model seed=$seed $L5 +shard=$i +n_shards=$POOL profile.num_workers=$WORKERS >> '$LOG' 2>&1"
      done
    done
  done
} | pool_run "$POOL"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    n=$(ls "$ART/pretrain/$model/${WINDOW}__${TAG}/seed$seed"/*.pt 2>/dev/null | wc -l)
    say "inits $model seed=$seed: $n/13"
    [ "$n" -eq 13 ] || { say "ABORT: $model seed=$seed has $n/13"; exit 1; }
  done
done

# ---------------------------------------------------------------------------
# Step 3 -- fine-tuning, unchanged from Phase 2 except for the initialisation.
# ---------------------------------------------------------------------------
say "Step 3: per-patient fine-tuning"
{
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "python scripts/train.py profile=server data=chbmit model=$model seed=$seed $L5 profile.num_workers=$WORKERS >> '$LOG' 2>&1"
    done
  done
} | pool_run "$POOL"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    n=$(ls "$ART/$model/$SPLIT/${WINDOW}__${TAG}/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
    say "folds $model seed=$seed: $n/66"
    [ "$n" -eq 66 ] || { say "ABORT: $model seed=$seed has $n/66"; exit 1; }
  done
done

# ---------------------------------------------------------------------------
# Step 4 -- score on the same post-processing config as rows 32-34, so L5 is
# the only difference between the two arms.
# ---------------------------------------------------------------------------
say "Step 4: scoring"
{
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "python scripts/rethreshold.py profile=server data=chbmit model=$model seed=$seed train.run_tag=$TAG postprocess=hysteresis_widegrid >> '$LOG' 2>&1"
    done
  done
} | pool_run "$POOL"

for model in "${MODELS[@]}"; do
  say "evaluate: $model + L5, 3 seeds, v2 gates"
  run python scripts/evaluate.py profile=server data=chbmit model="$model" \
    train.run_tag="$TAG" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
    profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml
done

# ---------------------------------------------------------------------------
# Step 5 -- the measurement that matters: L5 against its own control, same
# architecture, same seeds, same post-processing. Anything else confounds the
# corpus change with an architecture change.
# ---------------------------------------------------------------------------
for model in "${MODELS[@]}"; do
  say "paired bootstrap: $model WITH L5 vs WITHOUT (its Phase 2 control)"
  run python scripts/paired_bootstrap.py \
    "$ART/$model/$SPLIT/${WINDOW}__${TAG}" "$ART/$model/$SPLIT/$WINDOW" \
    --all-metrics --json "$ART/paired_${model}_L5_vs_noL5.json"
done

say "Phase 3 done. Send back $LOG and the paired_*_L5_vs_noL5.json files"
