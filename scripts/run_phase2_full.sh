#!/usr/bin/env bash
# Phase 2: settle the architecture question properly.  Stage A ~2h, Stage B ~50h.
#
# Where this comes from
# ---------------------
# The probe (row 31) showed cohort pre-training lifts `frontiers2d` by 8.9pp, to
# 0.9705 -- past a gate 26 runs had recorded as unreachable -- while the same
# lever gives `k5only` 0.9358. So Phase 1's "k5only leads" was an artefact of
# comparing a pre-trained model against un-pre-trained baselines.
#
# But frontiers2d+L1 sits at FAR 0.3459/h against k5only's 0.2261/h, and
# worst-patient FAR 1.1677 against 0.7785. It is 1.5x more expensive in false
# alarms on both counts. Part of that 3.5pp sensitivity gap was therefore bought
# with false alarms, not earned -- and how much is unknown.
#
# Stage A answers that with no training at all. Stage B then supplies the error
# bars, because row 31 is a single seed and this project was burned by exactly
# that one day earlier: row 24's reported 0.9359 turned out to be the top of its
# own seed range, with a 3-seed mean 1.1pp BELOW the row it was said to beat.
#
# Resumable: rethreshold is idempotent, and training skips folds that already
# have a metrics.json. Run under nohup; restart after an interruption.
#
#   nohup bash scripts/run_phase2_full.sh > phase2.out 2>&1 &
set -uo pipefail

: "${CHBMIT_RAW_DIR:?set CHBMIT_RAW_DIR first}"
: "${WEARSEIZURE_ARTIFACTS_DIR:?set WEARSEIZURE_ARTIFACTS_DIR first}"

ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ART="$WEARSEIZURE_ARTIFACTS_DIR"
LOG="$ART/phase2_full.log"
SPLIT=patient_specific_loso_edf
WINDOW=w4s_stride1s

say() { printf '\n=== [%s] %s ===\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "Phase 2 start. commit=$(git rev-parse --short HEAD)"

# ===========================================================================
# STAGE A -- the sensitivity/FAR operating curve, at matched FAR (~2h)
#
# Comparing two models at whatever FAR their threshold search happened to land
# on is not a comparison. `far_cap_per_hour` bounds validation FAR during
# threshold selection, so sweeping it traces each model's operating curve; the
# comparison is then read off at equal TEST FAR rather than at equal cap.
#
# Fixed at the row 22 post-processing config, which is k5only's better one
# (0.9358 vs 0.9179) and which frontiers2d+L1 has never been run on.
#
# Reports are copied aside per (model, cap) because rethreshold overwrites the
# same *.metrics.json each pass.
# ===========================================================================
CAPS=(0.10 0.15 0.20 0.30)

sweep_far_cap() {                # sweep_far_cap <model> <seed...>
  local model="$1"; shift
  local seeds=("$@")
  local d="$ART/$model/$SPLIT/$WINDOW"
  for cap in "${CAPS[@]}"; do
    for seed in "${seeds[@]}"; do
      say "STAGE A rethreshold: model=$model seed=$seed far_cap=$cap"
      run python scripts/rethreshold.py profile=server data=chbmit \
        model="$model" seed="$seed" postprocess=hysteresis_widegrid \
        postprocess.far_cap_per_hour="$cap"
    done
    say "STAGE A evaluate: model=$model far_cap=$cap (${#seeds[@]} seed(s))"
    run python scripts/evaluate.py profile=server data=chbmit \
      model="$model" postprocess=hysteresis_widegrid \
      postprocess.far_cap_per_hour="$cap" \
      "train.seeds=[$(IFS=,; echo "${seeds[*]}")]" \
      profile.enforce_gates=false \
      eval.gates_path=configs/eval/gates_v2_proposed.yaml
    for f in report_multiseed.json "seed${seeds[0]}/report.json"; do
      [ -f "$d/$f" ] && cp "$d/$f" "$d/report_farcap${cap}_$(basename "$f")"
    done
  done
}

sweep_far_cap wearseizure1d_k5only 0 1 2
sweep_far_cap baseline_frontiers2d 0          # only seed 0 has L1 so far

# Summary goes to a SEPARATE file. `grep "$LOG" | tee -a "$LOG"` appends every
# matching line back into the file it just read, doubling them -- which silently
# breaks any later `grep -c` used to track progress.
say "STAGE A done. The comparison to read: sensitivity of each model at EQUAL test FAR."
grep -E "macro sensitivity|STAGE A evaluate" "$LOG" | tail -40 > "$ART/phase2_stageA_summary.txt"
say "stage A summary -> $ART/phase2_stageA_summary.txt"

# ===========================================================================
# STAGE B -- error bars for the L1 arm of every architecture (~50h)
#
# Row 31 is one seed. Before the paper's thesis moves from "our architecture
# detects better" to "cohort pre-training closes the gap, and ours does it at
# 4.3x lower compute", both baselines need the same three seeds k5only already
# has. compact1d_7k has never been run with L1 at all.
# ===========================================================================
for combo in "baseline_frontiers2d 1" "baseline_frontiers2d 2" \
             "baseline_compact1d_7k 0" "baseline_compact1d_7k 1" \
             "baseline_compact1d_7k 2"; do
  set -- $combo
  model="$1"; seed="$2"

  # compact1d_7k seed 0 already holds an un-pre-trained run (rows 1/3/5/7/13).
  # Preserve it the way the probe preserved frontiers2d's: the -L1 arm of the
  # ablation grid still needs those checkpoints, and they cost GPU hours.
  d="$ART/$model/$SPLIT/$WINDOW"
  if [ "$seed" = "0" ] && [ -d "$d/seed0" ] && [ ! -d "$d/seed0_noL1" ]; then
    say "preserving the existing no-L1 $model run as seed0_noL1"
    mv "$d/seed0" "$d/seed0_noL1"
    mkdir -p "$d/seed0"
  fi

  say "STAGE B inits: model=$model seed=$seed (3 shards)"
  for i in 0 1 2; do
    python scripts/pretrain_cohort.py profile=server data=chbmit \
      model="$model" seed="$seed" +shard="$i" +n_shards=3 profile.num_workers=4 \
      >> "$LOG" 2>&1 &
  done
  wait
  n_init=$(ls "$ART/pretrain/$model/$WINDOW/seed$seed"/*.pt 2>/dev/null | wc -l)
  say "inits for $model seed=$seed: $n_init/13"
  [ "$n_init" -eq 13 ] || { say "ABORT: expected 13 inits, found $n_init"; exit 1; }

  say "STAGE B training: model=$model seed=$seed"
  run python scripts/train.py profile=server data=chbmit \
    model="$model" seed="$seed" train.pretrain.enabled=true
  n_folds=$(ls "$d/seed$seed"/*.metrics.json 2>/dev/null | wc -l)
  say "folds for $model seed=$seed: $n_folds/66"
  [ "$n_folds" -eq 66 ] || { say "ABORT: expected 66 folds, found $n_folds"; exit 1; }
done

# All three architectures now have three L1 seeds. Score them on one shared
# configuration so the only thing differing is the architecture.
for model in wearseizure1d_k5only baseline_frontiers2d baseline_compact1d_7k; do
  for seed in 0 1 2; do
    run python scripts/rethreshold.py profile=server data=chbmit \
      model="$model" seed="$seed" postprocess=hysteresis_widegrid
  done
  say "STAGE B evaluate: $model, 3 seeds, v2 gates"
  run python scripts/evaluate.py profile=server data=chbmit \
    model="$model" postprocess=hysteresis_widegrid 'train.seeds=[0,1,2]' \
    profile.enforce_gates=false eval.gates_path=configs/eval/gates_v2_proposed.yaml
done

K5="$ART/wearseizure1d_k5only/$SPLIT/$WINDOW"
for other in baseline_frontiers2d baseline_compact1d_7k; do
  say "STAGE B paired bootstrap: k5only vs $other, 3 seeds each"
  run python scripts/paired_bootstrap.py "$K5" "$ART/$other/$SPLIT/$WINDOW" \
    --all-metrics --json "$ART/paired_k5only_vs_${other}_3seed.json"
done

say "Phase 2 done. Send back $LOG and the paired_*_3seed.json files"
