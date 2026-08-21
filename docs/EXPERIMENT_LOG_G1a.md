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

| 24 | 08-17 13:10 | wearseizure1d — L1 + lưới rộng + **`run_length=2`, `ema_alpha=0.25`** | w4s_stride1s | rethreshold only | **0.9359** ❌ (**tốt nhất từng đạt**) | **0.2610 ✅(minimum)** | **16.17** / — ❌ | — | — | 185.0 ✅(target) | Sàn thật **8.00 s** (log in 13.00 s — server còn chạy code trước `dde3f74`). Phần dư model 8.17 s. **Cấu hình tham chiếu mới**: trội hơn row 22 ở *cả* sensitivity (+1.4 pp) *và* delay (−0.9 s), FAR vẫn dưới trần 0.30 |
| 25 | 08-17 13:16 | wearseizure1d — L1 + lưới rộng + `run_length=1`, `ema_alpha=0.25` | w4s_stride1s | rethreshold only | 0.9114 ❌ | 0.2360 ✅(minimum) | 16.38 ❌ | — | — | 185.0 ✅(target) | Sàn thật 7.00 s, dư model 9.38 s. **Bị row 22 áp đảo** (thấp hơn cả sensitivity lẫn FAR) — quan hệ (L, α) không đơn điệu |
| 26 | 08-17 13:21 | wearseizure1d — L1 + lưới rộng + `run_length=2`, `ema_alpha=0.5` | w4s_stride1s | rethreshold only | 0.8731 ❌ | 0.5030 ❌ | 15.74 ❌ | — | — | 185.0 ✅(target) | Sàn thật 6.00 s, dư model 9.74 s. Bị áp đảo |

**Quét (run_length, ema_alpha) trên cùng checkpoint L1 — mặt Pareto:** chỉ row 22 và row 24 nằm
trên biên; rows 23/25/26 đều bị áp đảo. Điểm quan trọng: hạ sàn từ 13.0 xuống 5.0 s **chỉ mua được
2.7 s delay thật** (17.06 → 14.36), vì phần dư model tăng ngược từ 4.06 lên 9.36 s. EMA nặng đang
làm việc thật — nó tích luỹ bằng chứng nên alarm nổ gần như ngay khi trả xong sàn; bỏ nó thì model
phải tự tin trong một cửa sổ đơn lẻ, điều xảy ra muộn hơn trong cơn. **Phép phân rã sàn/phần dư là
công cụ kế toán, không phải bất biến vật lý** — chỉ tổng delay mới so sánh được giữa các cấu hình.

**Cảnh báo về ý nghĩa thống kê:** ba cấu hình tốt nhất cách nhau 0.9218 / 0.9256 / 0.9359, tức
1.4 pp trên 77 cơn ≈ **đúng một cơn**. Chưa có thanh sai số nào (`train.seeds` vẫn là cấu hình
chết), nên **không thể khẳng định row 24 > row 22**. Lever L7 giờ là điều kiện chặn cho mọi so sánh
trong nhóm này.

### Provenance: which SERVER-02 run produced which row

Recovered from `.hydra/overrides.yaml` in `$WEARSEIZURE_ARTIFACTS_DIR/runs/` on 08-19, because
none of it was recorded anywhere in the repository. Rows 21-26 all share **one** set of 66
checkpoints (`train.py`, run `07-16-16`, `train.pretrain.enabled=true train.force_retrain=true`);
every later row is a `rethreshold.py` pass over those same checkpoints, so they differ only in
post-processing.

| Run directory | Overrides beyond `profile=server data=chbmit` | Row |
|---|---|---|
| `2026-08-17_07-16-16` train | `train.pretrain.enabled=true train.force_retrain=true` | L1 training for rows 21-26 |
| `2026-08-17_11-03-24` eval | none (default grid, `on_grid` floor 0.20) | **21** |
| `2026-08-17_11-54-01` → `12-23-46` | wide grid, `run_length=3`, `ema_alpha=0.125` | **22** |
| `2026-08-17_12-30-57` → `12-36-41` | wide grid, `run_length=1`, `ema_alpha=0.5` | **23** |
| `2026-08-17_13-04-49` → `13-10-23` | wide grid, `run_length=2`, `ema_alpha=0.25` | **24** |
| `2026-08-17_13-10-30` → `13-16-02` | wide grid, `run_length=1`, `ema_alpha=0.25` | **25** |
| `2026-08-17_13-16-08` → `13-21-32` | wide grid, `run_length=2`, `ema_alpha=0.5` | **26** |

The "wide grid" is now a committed, named config, `configs/postprocess/hysteresis_widegrid.yaml`,
rather than a command-line override that survives only in a gitignored directory:

```
on_grid:  [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
off_grid: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
```

**Row 22 was reproduced from the saved checkpoints on 08-19**, after the artifacts were migrated
into the new `seed<N>/` layout: sensitivity 0.9218, FAR 0.1878/h, delay median 13.00s, worst-patient
sensitivity 0.3333, worst-patient FAR 0.6182, exposure 185.0h — every one an exact match. A first
attempt using a hand-reconstructed grid (four extra points) matched sensitivity and FAR exactly but
gave delay mean 17.01s instead of 17.06s, one fold having picked a threshold one notch away. That
0.05s is the entire practical argument for committing the grid instead of retyping it.

Note that the directory `wearseizure1d/patient_specific_loso_edf/w4s_stride1s/` does **not** hold
row 22: each `rethreshold.py` pass overwrites the same `*.metrics.json`, so it holds row 26, the
last one run. Row 22 survives separately in `row22_backup/`.

Row 11 and row 17/12 are duplicates of the same underlying run (kept separate because they were reported to me at different points for different reasons — row 11 as the fd-crash diagnosis, row 17 as part of the 4-way kernel ablation).

## 2b. Phase 1 — the first results with error bars (rows 27-30, 08-20)

Three seeds per configuration (lever L7), two architectures, both post-processing
configurations. All from `scripts/run_phase1_server02.sh`; all share `postprocess=hysteresis_widegrid`.
Values are **mean +/- sample std across 3 seeds**, copied verbatim from the `[3 seeds]` lines of
`phase1_server02.log`.

| # | Model | Postproc | Sensitivity | FAR/h | Delay mean | Delay median | Floor | Worst-pt sens (>=5 seizures) | Worst-pt FAR/h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 27 | `wearseizure1d` | row22 cfg (L=3, a=.125) | 0.9175 ± 0.0171 | 0.2701 ± 0.0774 | 18.20 ± 1.20 | 14.33 | 13.0 | **0.9000 ± 0.0500** (chb15, 20 ev) | 0.8472 ± 0.2098 |
| 28 | **`k5only`** | **row22 cfg** | **0.9358 ± 0.0304** | **0.2261 ± 0.0601** | 18.83 ± 0.69 | 15.33 | 13.0 | **0.8714 ± 0.0247** | 0.7785 ± 0.2860 |
| 29 | `wearseizure1d` | row24 cfg (L=2, a=.25) | 0.9063 ± 0.0284 | 0.3406 ± 0.0972 | 16.94 ± 0.68 | 13.00 | 8.0 | 0.8190 ± 0.0330 | 1.1231 ± 0.4452 |
| 30 | `k5only` | row24 cfg | 0.9179 ± 0.0391 | 0.2496 ± 0.0687 | 17.42 ± 1.18 | 13.67 | 8.0 | 0.8000 ± 0.0000 (chb08, 5 ev) | 0.8701 ± 0.3246 |

Paired cluster bootstrap by patient, `k5only` vs `wearseizure1d`, on the **row 24** configuration
(the one left on disk when the driver finished), 10 000 replicates:

| Metric | k5only | default | delta | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| sensitivity_macro | 0.9179 | 0.9063 | +0.0115 | [-0.0256, +0.0579] | indistinguishable |
| sensitivity_micro | 0.9134 | 0.9177 | -0.0043 | [-0.0286, +0.0343] | indistinguishable |
| far_per_hour_micro | 0.2577 | 0.3658 | -0.1081 | [-0.2118, -0.0023] | **k5only better** |
| delay_mean_s | 17.4455 | 16.9292 | +0.5163 | [-0.3261, +1.2446] | indistinguishable |

(The verdict column was initially printed inverted for the two lower-is-better metrics; fixed in
commit `a13539e`, and the FAR line reads as a k5only win.)

### Three findings, two of which overturn earlier conclusions

**1. The worst-patient gate was never failing. It was measuring chb17.**

`worst_patient_sensitivity` sat at 0.3333 across 26 runs and was recorded as never cleared. Under
the `min_events_to_gate: 5` rule it lands on a patient where a sensitivity threshold is actually
expressible, and the picture reverses completely:

| Config | Gated patient(s) across seeds | Worst-pt sens |
|---|---|---:|
| row 27 | chb15 (20 ev) x3 | **0.9000** |
| row 28 | chb23 (7), chb03 (7), chb15 (20) | **0.8714** |
| row 29 | chb23 (7), chb08 (5), chb08 (5) | 0.8190 |
| row 30 | chb08 (5) x3 | 0.8000 |

Rows 27 and 28 clear **0.85 — the original v1 minimum**, and row 27 reaches the v1 *target* of 0.90.
The gate this project has been failing for its entire history was, in three of four configurations,
already passed. This is exactly the argument of section 4 of `RESEARCH_REALITY_CHECK.md`, now
measured rather than predicted.

The honest caveat is large and must be stated in the paper: **6 of 13 patients are exempt** --
chb02, chb04, chb07, chb11, chb17, chb22 all have fewer than 5 seizures. Only 7 patients are
gateable at all. The exempt six are reported with exact binomial intervals in
`report["small_sample_patients"]`, and their raw worst value is still carried as
`worst_patient_sensitivity_all` (0.4444-0.6111 across these rows) so nothing is hidden.

**2. "Row 24 beats row 22" was seed noise. With error bars the ranking flips.**

Single-seed row 24 scored 0.9359 against row 22's 0.9218, and the log recorded row 24 as the "new
reference configuration". Across three seeds:

| Config | 3-seed mean | Single-seed value that was reported | Seed range |
|---|---:|---:|---|
| row 22 cfg, default | **0.9175** | 0.9218 | 0.8987 - 0.9321 |
| row 24 cfg, default | **0.9063** | 0.9359 | 0.8800 - 0.9364 |

Row 24's 0.9359 was the top of its own seed range -- the favourable draw. Its mean is 1.1pp *below*
row 22's. The seed spread is 3-6pp, i.e. **two to four times the 1.4pp gap that was being ranked
on**. Every ordering of rows 22-26 by point estimate should be treated as unsupported.

**3. The row 22 configuration is the better operating point, not row 24.**

For `k5only`, row 22 cfg dominates row 24 cfg on both axes at once: sensitivity 0.9358 vs 0.9179 and
FAR 0.2261 vs 0.2496. The cost is the delay floor (13.0s vs 8.0s) and 1.4s of real delay
(18.83 vs 17.42). Since `detection_delay_floor_s` is a constraint on the measurement setup rather
than on the model, and the mean delay difference is small, row 22 cfg is the configuration to carry
forward.

Consequence for the architecture comparison: the paired bootstrap above was run on row 24 cfg, the
weaker configuration for both models. It needs re-running on row 22 cfg -- `rethreshold` only, no
training -- before the architecture question is settled.

## 2c. Row 31 — the probe that killed the architecture claim (08-21)

`scripts/run_phase2_probe.sh`: **`baseline_frontiers2d` WITH cohort pre-training**, seed 0, row 24
post-processing config (L=2, a=0.25), same wide threshold grid as rows 27-30. One combination,
~5h, run to answer one question before committing ~60h to the full grid.

| # | Model | L1 | Seeds | Sensitivity | FAR/h | Delay mean | Floor | Model reaction |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2 | `frontiers2d` | no | 1 | 0.8811 | 0.1432 | 23.40 | 13.0 | 10.40 |
| 31 | **`frontiers2d`** | **yes** | 1 | **0.9705** | 0.3297 (micro) / 0.346 (macro) | **13.16** | 8.0 | **5.16** |
| 30 | `k5only` | yes | 3 | 0.9179 ± 0.0391 | 0.2577 (micro) | 17.42 ± 1.18 | 8.0 | ~9.42 |
| 28 | `k5only` | yes | 3 | 0.9358 ± 0.0304 | 0.2261 | 18.83 ± 0.69 | 13.0 | ~5.83 |

Paired bootstrap, `k5only`+L1 (3 seeds) vs `frontiers2d`+L1 (1 seed), row 24 config both sides:

| Metric | k5only | frontiers2d | delta | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| sensitivity_macro | 0.9179 | 0.9705 | -0.0526 | [-0.1317, +0.0185] | indistinguishable |
| sensitivity_micro | 0.9134 | 0.9740 | -0.0606 | [-0.1178, -0.0051] | **frontiers2d better** |
| far_per_hour_micro | 0.2577 | 0.3297 | -0.0721 | [-0.2129, +0.0599] | indistinguishable |
| delay_mean_s | 17.4455 | 13.1600 | +4.2855 | [+1.5659, +7.0755] | **frontiers2d better** |

### What this means

**Cohort pre-training, not architecture, is what closes the single-channel gap.** L1 lifts
`frontiers2d` by **8.9pp**, from 0.8811 to 0.9705 -- past the v1 gate of 0.970 that 26 runs had
recorded as unreachable. Applied to `k5only` the same lever gives 0.9358. The Phase 1 conclusion
that `k5only` leads was an artefact of comparing a pre-trained model against un-pre-trained
baselines.

The paper's thesis therefore changes. Not "this architecture detects better", but: cohort
pre-training closes the gap, and `k5only` reaches a *nearby* operating point at **4.3x lower
compute** (585,920 vs 2,523,328 MACs) -- an accuracy/compute trade-off, quantified, with the
accelerator built for the efficient end of it. The compute side needs no new experiment; it is
already measured.

### Three reasons not to rewrite anything yet

**1. Row 31 is ONE seed, and this log has just been burned by exactly that.** Row 24's 0.9359 was
the top of its own seed range and its 3-seed mean came out 1.1pp *below* row 22's. `k5only`'s own
single-seed maxima are 0.9577 (row 24 cfg) and 0.9621 (row 22 cfg) -- so a favourable draw can move
a number by 4pp here. 0.9705 from one seed cannot carry a thesis change.

**2. The comparison is not at matched FAR.** `frontiers2d`+L1 sits at FAR 0.3297/h while `k5only`
sits at 0.2577/h. Part of that 5.3pp sensitivity gap was bought with false alarms, and how much is
unknown until both are re-thresholded to the same FAR. This costs a `rethreshold` pass, no training.

**3. `k5only` was measured on its weaker configuration.** Row 22 cfg gives it 0.9358, not 0.9179,
and `frontiers2d`+L1 has never been run on row 22 cfg at all.

None of these three requires retraining except the seeds.

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
