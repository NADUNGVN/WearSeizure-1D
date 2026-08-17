# WearSeizure-1D — Experiment Log for Gate G1a (Personalized/Patient-Specific, CHB-MIT)

Compiled for external review. All numbers below are copied verbatim from actual
`scripts/evaluate.py` / `scripts/rethreshold.py` / `scripts/failure_analysis.py`
log output on SERVER-02 (real CHB-MIT data, 13 patients, 66 EDF-file folds under
`patient_specific_loso_edf`), unless marked "synthetic" or "not reported". No
numbers were estimated or interpolated.

## 1. Gate thresholds (`configs/eval/gates.yaml`, Table 6 of the Research Decision Memo)

Personalized / patient-specific mode (the mode all experiments below target):

| Gate | Minimum | Target | Stretch | Direction |
|---|---|---|---|---|
| `personalized_event_sensitivity` | 0.970 | 0.985 | 0.990 | higher is better |
| `far_per_hour` | 0.30 | 0.20 | 0.10 | lower is better |
| `detection_delay_mean_s` | 5.0 | 4.0 | 3.0 | lower is better |
| `detection_delay_median_s` | (none) | 3.0 | 2.0 | lower is better |
| `worst_patient_sensitivity` | 0.85 | 0.90 | 0.95 | higher is better |
| `worst_patient_far_per_hour` | 1.0 | 1.0 | 0.5 | lower is better |
| `continuous_test_exposure_hours` | 100 | 150 | 200 | higher is better |

Zero-shot LOSO mode (not yet exercised in any run below — all runs so far are `patient_specific_loso_edf`):

| Gate | Minimum | Target | Stretch |
|---|---|---|---|
| `zero_shot_loso_sensitivity` | 0.85 | 0.90 | 0.92 |
| `zero_shot_loso_far_per_hour` | 0.75 | 0.50 | 0.30 |

Model budget (hard ceiling enforced by unit tests, independent of Gate G1a):

| Gate | Minimum | Target | Stretch |
|---|---|---|---|
| `model_params` | 32,000 | 16,000 | 12,000 |
| `model_macs` | 2,000,000 | 700,000 | 500,000 |

`wearseizure1d` (Table 4 design intent, not a measured/profiled count): **13,810 params / 644,000 MACs**. Actual measured params/MACs for any variant were never printed in this session — see Gaps section.

## 2. Master results table — all real-data (CHB-MIT) runs, chronological

All rows: split = `patient_specific_loso_edf`, dataset exposure ≈ 185.0h (66 folds, 13 patients) unless noted otherwise. ✅/target/stretch/❌ reflect the `level` field logged by `evaluate.py` itself (not my judgment).

| # | Timestamp | Model | Window | Postprocess variant | Sensitivity | FAR/h | Delay mean/median (s) | Worst-pt Sens | Worst-pt FAR/h | Exposure (h) | Status vs minimum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 08-14 19:13 | baseline_compact1d_7k | w4s_stride1s | default grid, 30 epoch | 0.8075 ❌ | 0.1394 ✅(target) | 25.04 / 22.0 ❌ | 0.0000 ❌ | 0.3798 ✅(stretch) | 185.0 ✅(target) | 2/6 minimums pass |
| 2 | 08-14 19:13 | baseline_frontiers2d | w4s_stride1s | default grid, 30 epoch | 0.8811 ❌ | 0.1432 ✅(target) | 23.40 / 19.0 ❌ | 0.6667 ❌ | 0.6903 ✅(target) | 185.0 ✅(target) | 3/6 minimums pass |
| 3 | 08-14 19:54 | baseline_compact1d_7k | w4s_stride1s | widened on_grid down to 0.0 | 0.8808 ❌ | 0.2612 ✅(minimum) | 18.11 / 13.0 ❌ | 0.3333 ❌ | 1.0355 ❌ | 185.0 ✅(target) | 2/6 minimums pass |
| 4 | 08-14 19:54 | baseline_frontiers2d | w4s_stride1s | widened on_grid down to 0.0 | 0.9031 ❌ | 0.2855 ✅(minimum) | 19.92 / 16.0 ❌ | 0.6667 ❌ | 1.3807 ❌ | 185.0 ✅(target) | 2/6 minimums pass |
| 5 | 08-15 02:11 | baseline_compact1d_7k | w4s_stride1s | far-capped constrained selection (≤0.30 val FAR, max sens) | 0.8808 ❌ | 0.2612 ✅(minimum) | 18.32 / 13.0 ❌ | 0.3333 ❌ | 1.0355 ❌ | 185.0 ✅(target) | bit-for-bit identical to #3 — confirms far_cap wasn't binding |
| 6 | 08-15 02:11 | baseline_frontiers2d | w4s_stride1s | far-capped constrained selection | 0.8921 ❌ | 0.2855 ✅(minimum) | 20.13 / 16.0 ❌ | 0.6667 ❌ | 1.3807 ❌ | 185.0 ✅(target) | ~identical to #4 |
| 7 | 08-15 10:16 | baseline_compact1d_7k | w4s_stride1s | 60 epoch (was 30), patience 15 (was 8) | 0.8974 ❌ | 0.2103 ✅(minimum, near target) | 19.49 / 18.0 ❌ | 0.3333 ❌ | 0.5697 ✅(target) | 185.0 ✅(target) | 3/6 minimums pass |
| 8 | 08-15 13:44 | baseline_frontiers2d | w4s_stride1s | 60 epoch, patience 15 | 0.9185 ❌ | 0.4061 ❌ (regressed vs 30-epoch's 0.285) | 19.52 / 17.0 ❌ | 0.6667 ❌ | 2.2436 ❌ | 185.0 ✅(target) | 1/6 minimums pass — FAR got worse with more training |
| 9 | 08-15 13:55 | baseline_compact1d_7k | w4s_stride1s | aggressive: ema_alpha=0.5, run_length=1 | 0.9487 ❌ (closest sens yet) | 1.0240 ❌ (blew past minimum) | 14.67 / 12.0 ❌ | 0.6667 ❌ | 2.7614 ❌ | 185.0 ✅(target) | 1/6 — classic sens/FAR trade-off, too aggressive |
| 10 | 08-15 14:05 | baseline_compact1d_7k | w4s_stride1s | moderate: ema_alpha=0.25, run_length=2 | 0.9377 ❌ | 0.4975 ❌ | 18.69 / 16.5 ❌ | 0.6667 ❌ | 1.3807 ❌ | 185.0 ✅(target) | 1/6 |
| 11 | 08-15 17:51 | wearseizure1d (default) | w4s_stride1s | 30 epoch | 0.9152 ❌ | 0.3263 ❌ | 18.54 / 14.0 ❌ | 0.5000 ❌ | 1.5533 ❌ | 162.4 ✅(target) | **partial run — only 59/66 folds** (crashed on "Too many open files" mid-run); not representative, superseded by #12 |
| 12 | 08-15 18:13 | wearseizure1d (default) | w4s_stride1s | 30 epoch, full 66 folds (after fd-leak fix) | 0.8806 ❌ | 0.3119 ❌ | 19.42 / 16.0 ❌ | 0.5000 ❌ | 1.5533 ❌ | 185.0 ✅(target) | 1/6 |
| 13 | 08-15 23:31 | baseline_compact1d_7k | **w4s_stride0p5s** (finer stride ablation) | 60 epoch, ema_alpha=0.25/run_length=2 threshold | 0.8773 ❌ | 0.2775 ✅(minimum) | 18.37 / 13.0 ❌ | 0.5000 ❌ | 0.6595 ✅(target) | 185.0 ✅(target) | 3/6 — stride change alone didn't beat baseline |
| 14 | 08-16 10:19 | **wearseizure1d_k3only** (kernel ablation A) | w4s_stride1s | shared 60-ep/threshold config | 0.8396 ❌ | 0.3783 ❌ | 23.82 / 18.5 ❌ | 0.2500 ❌ | 2.5888 ❌ | 185.0 ✅(target) | 0/6 — worst variant |
| 15 | 08-16 10:19 | **wearseizure1d_k5only** (kernel ablation B) | w4s_stride1s | shared 60-ep/threshold config | 0.8756 ❌ | **0.1624 ✅(target)** | 23.46 / 16.5 ❌ | 0.2500 ❌ | **0.3926 ✅(stretch)** | 185.0 ✅(target) | 2/6, but FAR-side gates are at target/stretch — best FAR profile found |
| 16 | 08-16 10:19 | **wearseizure1d_nodilation** (kernel ablation C) | w4s_stride1s | shared 60-ep/threshold config | 0.8698 ❌ | 0.2250 ✅(minimum) | 23.14 / 19.0 ❌ | 0.5000 ❌ | 1.3807 ❌ | 185.0 ✅(target) | 2/6 |
| 17 | 08-16 10:19 | **wearseizure1d** default (kernel ablation D, multi_scale+dilation) | w4s_stride1s | shared 60-ep/threshold config | 0.8806 ❌ (best sens of the 4) | 0.3119 ❌ | 19.42 / 16.0 ❌ | 0.5000 ❌ | 1.5533 ❌ | 185.0 ✅(target) | 1/6 — same run as #12 |
| 18 | 08-16 16:31 | wearseizure1d_k5only | w4s_stride1s | rethreshold pushed lower: ema_alpha=0.25, run_length=2 | 0.8833 ❌ | 0.4083 ❌ | 20.79 / 13.5 ❌ | 0.2500 ❌ | 1.5533 ❌ | 185.0 ✅(target) | 1/6 — pushing threshold down traded away B's FAR advantage |
| 19 | 08-16 16:59 | wearseizure1d_k5only — **"Nhánh B": pooled per-patient threshold** | w4s_stride1s | per-fold val evidence pooled by patient (`rethreshold_pooled.py`) | **0.8236 ❌ (worse than #15's 0.876)** | 0.0621 ✅(stretch) | 26.86 / 20.5 ❌ (worse) | **0.0000 ❌ (worse than #15's 0.25)** | 0.1899 ✅(stretch) | 185.0 ✅(target) | **Negative result — pooling hurt sensitivity and worst-patient sensitivity despite improving FAR.** See §5. |
| 20 | 08-16 19:59 | wearseizure1d_k5only — **"Nhánh A": window=w2s_stride1s** | w2s_stride1s | default postprocess config (unchanged — still tuned for the 4s window) | 0.8749 ❌ (~same as #15's 0.8756) | 0.2919 ✅(minimum only — **worse** than #15's 0.1624 target) | **26.18 / 19.0 ❌ (worse** than #15's 23.46/16.5 — delay got *longer*, not shorter) | 0.2500 ❌ (same as #15) | **1.7259 ❌ (worse** than #15's 0.3926 stretch — blew past the 1.0 minimum) | 185.0 ✅(target) | 1/6 — **negative result**: shrinking the window did not reduce detection delay and regressed both FAR gates. See §5 step 8. |

| 21 | 08-17 11:03 | wearseizure1d — **L1: cohort pre-training** (13 inits, lr 1e-4 fine-tune) | w4s_stride1s | default grid (on_grid floor 0.20) | 0.8485 ❌ | **0.1243 ✅(target)** | 18.10 / 15.0 ❌ | 0.0000 ❌ | **0.3434 ✅(stretch)** | 185.0 ✅(target) | FAR/delay both improve vs #17, sensitivity falls 3.2pp — but FAR lands at 0.124 against a 0.30 cap, i.e. 2.4x of budget left unspent |
| 22 | 08-17 12:23 | wearseizure1d — **L1 + widened threshold grid** (`on_grid` down to 0.02) | w4s_stride1s | rethreshold only, no retraining | **0.9218** ❌ | **0.1878 ✅(target)** | **17.06 / 13.00** ❌ | 0.3333 ❌ (chb17, 3 events) | **0.6182 ✅(target)** | 185.0 ✅(target) | **Best result to date.** Strictly dominates `compact1d_7k` #7 (0.8974 @ 0.2103) on sensitivity *and* FAR, and beats `frontiers2d` #2 (0.8811) by 4.1pp. Model reaction 6.42 → 4.06s (−37%). Median delay 13.00s **equals the protocol floor exactly** |

| 23 | 08-17 12:36 | wearseizure1d — L1 + lưới rộng + **hậu xử lý nới: `run_length` 3→1, `ema_alpha` .125→.5** | w4s_stride1s | rethreshold only | **0.9256** ❌ | **0.7200 ❌ (vỡ trần 0.30)** | 14.36 / 11.00 ❌ | **0.6667** ❌ (tốt nhất từng đạt) | **2.2000 ❌** | 185.0 ✅(target) | Sàn thật **5.00 s** (không phải 13.00 s như log in ra — `evaluate.py` khi đó lấy sàn từ config thay vì từ tham số đã đóng băng; đã sửa). Phần dư model thật = 14.36 − 5.00 = **9.36 s**. Delay và worst-patient sensitivity đều tốt nhất, nhưng FAR đắt gấp 3.8× so với row 22 → **row 22 vẫn là cấu hình tốt nhất tổng thể** |

Row 11 and row 17/12 are duplicates of the same underlying run (kept separate because they were reported to me at different points for different reasons — row 11 as the fd-crash diagnosis, row 17 as part of the 4-way kernel ablation).

## 3. Per-patient failure notes (from `failure_analysis.py`, 08-15 23:41, w4s_stride1s, 60-epoch checkpoints)

Cohort mean segment AUROC: compact1d_7k 0.922, frontiers2d 0.937, wearseizure1d 0.909 (all across the same 66 folds).

| Patient | Flagged by | Root-cause pattern |
|---|---|---|
| **chb07** | All 3 models (compact1d_7k, frontiers2d, wearseizure1d) | Model-independent hard case. Fold `chb07_12` has segment AUROC 0.28–0.53 across all 3 architectures (vs cohort mean ~0.92) — genuinely harder signal, not a threshold artifact. |
| **chb04** | frontiers2d, wearseizure1d (not compact1d_7k) | Fold `chb04_28` has segment AUROC 0.38–0.43 (vs cohort mean ~0.92–0.94) in the 2 models that flag it — same "hard event" pattern as chb07_12. |
| **chb11** | All 3 models | Flagged for **FAR**, not sensitivity — segment AUROC is actually excellent here (0.92–0.98, well above cohort mean). This is a methodology/exposure artifact: very short test exposure for this patient makes FAR/h estimates noisy, not a genuine model weakness. |
| **chb17** | compact1d_7k, frontiers2d, wearseizure1d (varies: FAR flag for compact1d_7k, sensitivity flag for the other two) | Only 3 seizure events total → each of its 3 folds has exactly 1 validation event, making per-fold `val_sensitivity` extremely noisy (frequently 0.0 even at the lowest threshold-grid value). Root cause traced (line 1587–1601 of the transcript) to `threshold_on` never being reachable below the grid floor (0.5) originally — grid was later widened, but the small-sample noise itself remains. |
| **chb23** | frontiers2d, wearseizure1d (not compact1d_7k) | FAR flag; segment AUROC very high (0.97–0.99), similar exposure-artifact pattern to chb11. |
| **chb02** | compact1d_7k only | Sensitivity flag, n_events=3, moderate AUROC (0.75–0.99 across its 3 folds, one fold at 0.752). |

`worst_patient_sensitivity` in the aggregate report has repeatedly resolved to **chb17** (compact1d_7k) landing on exactly 1/3 = 0.3333333... across two different training runs (30-epoch and 60-epoch) — flagged in the transcript as strong evidence this is a fixed data/pipeline characteristic of that specific patient/event, not something more training fixes.

## 4. Gaps / not yet measured

- **Zero-shot LOSO mode** (`split=zero_shot_loso_subject`) — never run in this session; only `patient_specific_loso_edf` has real-data results.
- **Actual measured param count / MAC count** for any model variant — never printed via `thop`/profiler in this session. Only the *design targets* (wearseizure1d: 13,810 params / 644,000 MACs per Table 4) and the *budget ceilings* (32,000 params / 2,000,000 MACs, shared by all variants and enforced by unit tests) are known. k3only/k5only/nodilation should each have strictly fewer params/MACs than the default (fewer conv branches), but no exact numbers exist yet.
- **int8/PTQ/QAT quantization-loss gates** (`int8_loss_vs_fp32_pp`, `w4a8_loss_vs_fp32_pp`) — code exists (`quant/ptq.py`, `quant/qat.py`, `export_int8_reference.py`) but no run in this session produced numbers against these gates.
- **Nhánh B's k3only/nodilation/frontiers2d/compact1d_7k pooled-threshold behavior** — pooling was only tried on `wearseizure1d_k5only`; whether the same degradation pattern holds for other model variants is untested.
- **Branch A with re-tuned postprocessing** — row 20 reused the postprocess grid/hysteresis config that was tuned for the 4s window verbatim; whether a 2s window helps delay *at all* when paired with a postprocess config re-tuned for the shorter window (e.g. tighter `on_grid`/`off_grid`, shorter run-length) is still untested. The negative result in row 20 conflates "does a shorter window help" with "does a shorter window help under a stale, mismatched postprocess config" — these have not been separated yet.

## 5. Narrative timeline

1. First real-data baselines (compact1d_7k, frontiers2d) landed with FAR and exposure already at/above target, but sensitivity ~5–8pp short of the 97% minimum, detection delay 4–5x over the 5s minimum, and worst-patient sensitivity failing outright (one patient at 0%). Per-patient dump showed the 0% patient (chb17) never fired an alarm at all — a different failure mode from the "detects late" pattern seen elsewhere.
2. Root-caused to the fixed threshold-selection grid floor (0.5) being unreachable for low-confidence folds; widened the grid → sensitivity/delay improved but worst-patient FAR blew past minimum (classic sensitivity/FAR trade-off), confirming the diagnosis but not fixing the underlying issue.
3. Switched to a FAR-capped constrained threshold objective (max sensitivity subject to val FAR ≤ 0.30/h) — produced bit-for-bit identical results to the widened-grid version, revealing the cap was never actually binding; concluded that threshold-selection tuning had hit its ceiling and remaining gaps reflect real classifier limitations.
4. Doubled training (30→60 epochs, patience 8→15): modest sensitivity/FAR gains for compact1d_7k, but frontiers2d's FAR *regressed* (0.285→0.406) — more training is not a uniform win. `worst_patient_sensitivity` landed on the exact same value (0.3333...) in both the 30- and 60-epoch runs, pointing at a fixed per-patient data characteristic (chb17) rather than an undertrained model.
5. Tried aggressive/moderate hyper-aggressive threshold pushes (ema_alpha/run_length sweeps) purely as post-processing — got sensitivity up to 94.9% at best, but FAR always blew far past the minimum in exchange; no configuration cleared all 6 gates simultaneously.
6. Pivoted to architecture ablation per memo §7.2 (kernel_mode: k3_only / k5_only / nodilation / multi_scale+dilation, all under the wearseizure1d family). Found dilation contributes real sensitivity gain (D>C), and — the standout finding — the k3 branch is the dominant source of false alarms: k5_only alone loses only 0.5pp sensitivity vs the full default but nearly halves FAR (reaching target tier) and cuts worst-patient FAR ~4x (reaching stretch tier).
7. Explored two parallel "branches" from the k5_only result: **Branch A** (window=w2s_stride1s, to attack detection delay) first crashed on a config bug (model input_len hardcoded independent of window_s); bug fixed in `models/factory.py` and the run was redone against real data (row 20). **Branch B** (pool per-fold validation evidence by patient before selecting threshold, to attack worst-patient sensitivity) completed and produced a **negative result**: sensitivity dropped (87.6%→82.4%) and worst-patient sensitivity got worse, not better (25%→0%), because the "hard" patients have uniformly low-confidence validation across *all* their folds (not just noisy small-sample folds) — pooling more data doesn't help when every fold agrees the model isn't confident.
8. **Branch A's real-data result (row 20) is also a negative result**, and a more surprising one: shrinking the window to 2s did *not* shorten detection delay (mean 23.46s→26.18s, i.e. it got worse) and it regressed both FAR gates sharply (`far_per_hour` 0.162 target-tier → 0.292 barely-minimum; `worst_patient_far_per_hour` 0.393 stretch-tier → 1.726, blowing past the 1.0 minimum). Sensitivity and worst-patient sensitivity were essentially unchanged. The run reused the postprocess/hysteresis config that was tuned for the 4s window without re-tuning it for the shorter window — so this result does not yet distinguish "shorter window is a bad idea" from "shorter window needs its own postprocess tuning" (see Gaps).
9. As of the last entry in this log, no single configuration across 20 real-data experiments has cleared all 6 Table-6 minimum gates simultaneously. `far_per_hour`, `worst_patient_far_per_hour`, and `continuous_test_exposure_hours` have each been cleared in multiple configs; `personalized_event_sensitivity`, `detection_delay_mean_s`/`detection_delay_median_s`, and `worst_patient_sensitivity` have never been cleared in any run. Of the two parallel branches launched to attack the two hardest-remaining gates (delay, worst-patient-sensitivity), **both came back negative as tested** — the best real result overall remains row 15 (`wearseizure1d_k5only`, `w4s_stride1s`, default postprocess).
