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

## 2d. Phase 2 — three architectures, one recipe, three seeds (rows 32-34, 08-23)

All with cohort pre-training (L1), `postprocess=hysteresis_widegrid`, three seeds, same 66 folds.
This is **block B of the paper's benchmark table**: the only comparison that is valid, because
protocol, post-processing and training recipe are identical and only the architecture differs.

| # | Architecture | Sensitivity | FAR/h | Delay mean | Worst-pt sens | Worst-pt FAR | MACs |
|---|---|---:|---:|---:|---:|---:|---:|
| 32 | `wearseizure1d_k5only` | 0.9358 ± 0.0304 | **0.2261** | 18.83 | 0.8714 | 0.7785 | **585,920** |
| 33 | `baseline_frontiers2d` | **0.9726 ± 0.0037** | 0.2922 | 17.22 | 0.9500 | 2.2210 | 2,523,328 |
| 34 | `baseline_compact1d_7k` | 0.9440 ± 0.0286 | 0.3353 | 16.97 | 0.9667 | 1.8317 | 1,398,928 |

Paired cluster bootstrap by patient, 10 000 replicates, `k5only` as A:

| Metric | vs `frontiers2d` | vs `compact1d_7k` |
|---|---|---|
| sensitivity_macro | -0.0368, CI [-0.0946, +0.0096] — indistinguishable | -0.0082, CI [-0.0454, +0.0299] — indistinguishable |
| sensitivity_micro | -0.0390, CI [-0.0802, **-0.0044**] — B better | -0.0260, CI [-0.0521, +0.0117] — indistinguishable |
| far_per_hour_micro | -0.0577, CI [-0.3495, +0.1077] — indistinguishable | -0.0955, CI [-0.3041, +0.0223] — indistinguishable |
| delay_mean_s | +1.6111, CI [-1.7061, +4.1155] — indistinguishable | +1.8649, CI [-1.0344, +3.9052] — indistinguishable |

### The false-alarm advantage was one patient

Rows 32-34 look as though `k5only` has a large worst-patient FAR advantage: 0.7785/h against
2.2210 and 1.8317. It does not. The bootstrap on that axis came back **degenerate** -- a CI bound
sitting exactly on the point estimate -- because a max over 13 clusters is close to a yes/no
question about whether one particular patient landed in the replicate. The paired per-patient table
shows what was really happening:

| Patient | n_ev | FAR `k5only` | FAR `frontiers2d` | | FAR `compact1d_7k` | |
|---|---:|---:|---:|---|---:|---|
| chb01 | 7 | 0.147 | 0.195 | k5 | 0.195 | k5 |
| chb02 | 3 | 0.443 | 0.190 | f2d | 0.380 | c7k |
| chb03 | 7 | 0.095 | 0.000 | f2d | 0.024 | c7k |
| chb04 | 4 | 0.103 | 0.118 | k5 | 0.221 | k5 |
| chb05 | 5 | 0.133 | 0.067 | f2d | 0.233 | k5 |
| chb07 | 3 | 0.095 | 0.016 | f2d | 0.079 | c7k |
| chb08 | 5 | 0.200 | 0.100 | f2d | 0.067 | c7k |
| chb10 | 7 | 0.345 | 0.119 | f2d | 0.286 | c7k |
| chb11 | 3 | 0.345 | 0.575 | k5 | 0.345 | tie |
| chb15 | 20 | 0.143 | 0.143 | tie | 0.143 | tie |
| chb17 | 3 | 0.111 | 0.055 | f2d | 0.555 | k5 |
| chb22 | 3 | 0.000 | 0.000 | tie | 0.000 | tie |
| **chb23** | 7 | **0.778** | **2.221** | k5 | **1.832** | k5 |

Exact two-sided sign test: `k5only` is quieter in **4 of 11** patients against `frontiers2d`
(p = 0.55) and **5 of 10** against `compact1d_7k` (p = 1.00). On sensitivity, 1 of 6 (p = 0.22) and
2 of 6 (p = 0.69).

**The entire worst-patient FAR gap is chb23.** On the other twelve patients `frontiers2d` is more
often the quieter model. `k5only` has no false-alarm advantage, and no claim resting on one should
be made. This is the same failure mode `tests/unit/test_paired_bootstrap.py::
test_an_edge_carried_by_one_patient_is_not_called_significant` exists to catch -- it was caught by
looking at the distribution rather than at a summary of it.

### What Phase 2 actually established

1. **Cohort pre-training is the effect.** It lifts `frontiers2d` 0.8811 -> 0.9726, +8.9pp, with a
   seed std of 0.0037. That is the largest and most reproducible result in the project, and it is
   architecture-agnostic.
2. **The three architectures are statistically indistinguishable** on macro sensitivity, FAR and
   delay. `frontiers2d` wins micro sensitivity, marginally (CI upper bound -0.0044).
3. **`k5only` costs 4.3x and 2.4x fewer MACs** at that parity. Measured, not estimated. This is the
   only axis on which the architectures genuinely separate.

An honest caveat on (2): "indistinguishable" at 13 patients is partly a statement about **power**.
The macro point estimates differ by 3.7pp and micro reaches significance, so the correct phrasing is
that this cohort cannot separate them -- not that they are known to be equal.

## 2e. Rows 35-38 — lever L5 is a negative result, at both corpus widths (08-26 to 08-29)

Cohort pre-training corpus widened from the 12 remaining evaluation patients (553h) to those plus
the 11 non-evaluation CHB-MIT cases at all four wearable positions (2085h, **3.7x**). Evaluation
untouched: 13 cases, 66 folds, 185.0h. Three seeds, both architectures, `hysteresis_widegrid`,
each compared against **its own** Phase 2 control -- same architecture, same seeds, same
post-processing, corpus the only difference.

| # | Model | Pre-training corpus | Sensitivity | FAR/h |
|---|---|---|---:|---:|
| 32 | `k5only` | 13 cases (control) | **0.9358 ± 0.0304** | **0.2261** |
| 35 | `k5only` | + 11 cases, 4 positions | 0.9348 ± 0.0153 | 0.2318 |
| 37 | `k5only` | + 11 cases, **1 position** | **0.8924 ± 0.0116** | 0.2547 |
| 33 | `frontiers2d` | 13 cases (control) | **0.9726 ± 0.0037** | **0.2922** |
| 36 | `frontiers2d` | + 11 cases, 4 positions | 0.9641 ± 0.0170 | 0.4011 |
| 38 | `frontiers2d` | + 11 cases, **1 position** | 0.9641 ± 0.0279 | 0.3573 |

**The control wins all six cells.** Macro sensitivity and macro FAR, three seeds each, same
post-processing, same folds -- the only difference is what the cohort initialisation was
pre-trained on.

Paired bootstrap against each configuration's own control:

| Comparison | Δ sensitivity_macro | 95% CI | Verdict |
|---|---:|---|---|
| `k5only` + 4 positions | -0.0010 | [-0.0476, +0.0361] | indistinguishable |
| `k5only` + **1 position** | **-0.0434** | **[-0.0861, -0.0086]** | **control better** |
| `frontiers2d` + 4 positions | -0.0085 | [-0.0256, +0.0000] | indistinguishable (boundary) |

(Row-35/36 figures previously quoted here were the bootstrap's *micro* FAR; the table now uses
macro throughout, matching rows 32-34. The `report_multiseed.json` files were also being written to
the untagged control directory -- see commit `b237cfa`. The measurements were never affected, since
`paired_bootstrap.py` reads `*.metrics.json` directly and those were untouched.)

All twelve paired comparisons return "neither" -- no metric moves significantly, for either
architecture. Sign tests agree: L5 is quieter in 4 of 10 patients for `k5only` (p = 0.75) and 4 of
11 for `frontiers2d` (p = 0.55).

### Two things this establishes

**1. The pre-training benefit saturates at the 13-case cohort.** Lever L1 -- going from one
patient's recordings to twelve patients' -- was worth +6.0pp to `k5only` and +9.2pp to
`frontiers2d`, the largest effect in the project. A further 3.7x of the same kind of data is worth
nothing. The gain came from escaping a starved regime (a ~12k-parameter CNN learning from two
seizures), not from data volume as such, and that escape has already happened.

**2. Narrowing to one electrode position makes it WORSE, which reverses the diagnosis.**
The follow-up pre-registered in `configs/data/chbmit.yaml` was to narrow `pretrain_channels` to a
single position, on the theory that taking all four labels ictal windows positive at positions where
the seizure is not visible. It does the opposite of rescuing L5: `k5only` drops to 0.8924, and that
drop **is** significant (CI [-0.0861, -0.0086]).

The reasoning was backwards. With four positions at least one view has a chance of containing the
discriminative pattern; forcing every case onto P8-O2 means every patient whose seizure is not
visible there contributes *only* mislabelled ictal windows. A single fixed position raises the
mislabelled fraction rather than lowering it.

So the conclusion is stronger and simpler than "label noise": **the pre-training corpus cannot be
extended beyond the 13 clinically-confirmed cases at all.** Chung et al. excluded those 11 cases
precisely because seizure onset was not confirmed observable from these wearable positions, and this
is that exclusion being independently re-derived from the data.

The practical consequence is large and cheap to have bought: **external corpora (TUSZ, Siena) are
not worth pursuing.** They face the same problem at greater scale -- no per-patient confirmation of
which single channel carries the seizure -- and two runs of about a day each have settled what would
otherwise have been weeks of work.

**3. The extra corpus also degrades FAR, at both widths.**
`frontiers2d` degrades on exactly the axes label noise would touch: FAR 0.2793 -> 0.3928 (+41%) and
worst-patient FAR 2.2210 -> 3.3887 (+53%), with sensitivity almost unchanged (upper CI bound
exactly 0.0000). The 11 added cases have **no clinical confirmation** that seizure onset is
observable from these electrode positions -- that confirmation is precisely why Chung et al.
restricted to 13 cases -- and all four positions were taken per case. Ictal windows at a position
where the seizure is not visible are labelled positive regardless, teaching the model to fire on
patterns that are not seizures.

This was written down as the predicted failure mode in `configs/data/chbmit.yaml` before the run,
along with the follow-up: narrow `data.pretrain_channels` to a single position to separate "more
data does not help" from "more noisy data hurts".

### What it means for the paper

A bounded negative result on a lever that looked obvious, measured against its own control with
error bars. It also settles a forward-looking question cheaply: **external corpora (TUSZ, Siena)
are not worth pursuing on this evidence** unless the single-position ablation shows the loss was
noise rather than saturation. That would have been a much larger investment to discover the same
thing.

Note the point estimates for `k5only` all move the right way -- delay 18.83 -> 17.71, worst-patient
FAR 0.7785 -> 0.6640 -- but none of them significantly, and nothing should be built on them.

## 2f. Rows 39-40 — lever L4 changes the operating point, it does not improve it (08-30)

Checkpoint selection and early stopping on validation **AUPRC** instead of cross-entropy. Three
seeds, both architectures, cohort pre-training on, same post-processing, each compared against its
own control.

| # | Model | Selection | Sensitivity | FAR/h | Delay | Worst-pt FAR |
|---|---|---|---:|---:|---:|---:|
| 32 | `k5only` | val_loss | 0.9358 ± 0.0304 | 0.2261 | 18.83 | 0.7785 |
| 39 | `k5only` | **val_auprc** | 0.9380 ± 0.0350 | **0.1904** | 18.83 | 0.5701 |
| 33 | `frontiers2d` | val_loss | **0.9726 ± 0.0037** | **0.2922** | 17.22 | **2.2210** |
| 40 | `frontiers2d` | **val_auprc** | 0.9564 ± 0.0059 | 0.4673 | **15.03** | 4.1442 |

Paired bootstrap, L4 as A:

| Metric | vs `k5only` control | vs `frontiers2d` control |
|---|---|---|
| sensitivity_macro | +0.0022, CI [-0.0277, +0.0296] — neither | -0.0162, CI [-0.0385, +0.0000] — neither (boundary) |
| far_per_hour_micro | -0.0360, CI [-0.1241, +0.0310] — neither | **+0.1784, CI [+0.0057, +0.5371] — control better** |
| delay_mean_s | -0.0122, CI [-1.6112, +2.4462] — neither | **-2.1952, CI [-4.3039, -0.8909] — L4 better** |
| worst_patient_far | -0.3354, CI [-0.4121, +0.0357] — neither | +1.9233, degenerate — neither |

### What it did

Nothing measurable for `k5only`. For `frontiers2d` it moved the model along the operating curve
toward firing more readily: **2.2s faster and significantly so**, at **significantly worse FAR**,
with worst-patient FAR nearly doubling and sensitivity down 1.6pp. That is a different trade-off,
not a better one -- and it is the trade-off the wearable case cares least about, since detection
delay was already inside the clinical window while false alarms were the binding constraint.

### The confound, measured rather than assumed

**66 fold-trainings reached the 60-epoch ceiling**, against 312 that early-stopped. AUPRC keeps
improving after cross-entropy has saturated, so L4 also trains longer -- "selected on AUPRC" and
"trained longer" are therefore partly confounded. With a null result this matters less than it
would with a positive one, but any claim attributing an effect to the selection criterion has to
raise `train.epochs` first and re-run.

### Choosing versus claiming

`k5only` + L4 reaches **FAR 0.1904, the first configuration on record under the M3 threshold of
0.20**, with worst-patient FAR 0.5701 against M5's 0.50. Its interval contains zero, so it cannot
be *claimed* as an improvement -- but it can be *chosen* as the operating point, which needs no
significance test. Those are different acts and the paper should not blur them.

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

## 2g. Rows 41-43 -- two distillation levers that disagree (09-01 to 09-02)

Both arms ask whether the single-channel student can be taught by a stronger
model. They differ in ONE respect -- what the teacher reads -- and they come out
with opposite signs, which is the finding.

| # | student | teacher | teacher reads | sens macro | sens micro | FAR/h |
|---|---|---|---|---:|---:|---:|
| 32 | `k5only` | -- (control) | -- | 0.9358 | 0.9351 | 0.2216 |
| 41 | `k5only` | multi-channel (L3) | 18-26 channels | **0.9160** | 0.9221 | 0.2306 |
| 43 | `k5only` | `frontiers2d` (L8) | **the student's channel** | **0.9489** | **0.9567** | 0.2937 |
| 33 | `frontiers2d` | -- (control) | -- | 0.9726 | 0.9740 | 0.2793 |
| 42 | `frontiers2d` | multi-channel (L3) | 18-26 channels | **0.9513** | 0.9610 | **0.3658** |

Three seeds each, `hysteresis_widegrid`, gates v2, cluster bootstrap over 13
patients.

### L3 is negative, and for `frontiers2d` significantly so

- `k5only`: sens macro **-1.98pp** (CI [-6.03, +0.60], contains 0), FAR unchanged.
- `frontiers2d`: sens macro **-2.14pp** (CI [-5.13, 0.00]), and FAR micro
  **+0.0865/h**, CI [+0.037, +0.152], **excludes 0, favours the control**. The
  paired sign test agrees and is not degenerate: the control is quieter in
  **9 of 10** patients, exact two-sided **p = 0.021**.

This is the first lever measured to make things *significantly worse* on a
metric the paper reports, rather than merely failing to help.

### L8 moves the right way but does not clear the bar

- vs its own control: sens macro **+1.31pp** (CI [-2.52, +6.03]), micro
  **+2.16pp** (CI [-1.54, +5.45], p = 0.139). **Not significant.**
- vs the teacher it was distilled from: macro **-2.37pp**, CI
  **[-4.60, -0.51]**, and micro -1.73pp, CI [-3.86, -0.37]. **Both exclude 0 --
  the student is still significantly behind `frontiers2d`.**

The gap to the teacher narrowed from 3.68pp to 2.37pp, about a third of it, but
the part that closed does not clear the noise and the part that remains does.

### The mechanism the two arms establish together

The teacher that helped reads exactly what the student reads. The teacher that
hurt reads 18-26 channels the student does not have.

A soft target is only imitable if the student can, in principle, compute it. A
multi-channel teacher's confidence on a window depends on channels the student
cannot see, so on those windows the student is being trained toward a number it
has no way to derive -- and the KL term pulls it away from the hard labels for
nothing. That is consistent with lever L5, where extra pre-training data at the
WRONG electrode position was also worse than none.

So: **distillation helps when the teacher's advantage is capacity, and hurts
when it is information.** This is a methodological result that stands on its
own, and it is worth a subsection whichever way the remaining runs go.

### What L8 transferred that was not wanted

Per-patient, L8 is not a uniform lift:

| patient | control sens | L8 sens | control FAR | L8 FAR | teacher FAR |
|---|---:|---:|---:|---:|---:|
| chb04 | 0.5833 | **0.8333** | 0.103 | 0.000 | 0.118 |
| chb15 | 0.9000 | **0.9500** | 0.143 | 0.083 | 0.143 |
| chb07 | 0.8889 | **0.7778** | 0.095 | 0.111 | 0.016 |
| chb08 | 1.0000 | **0.9333** | 0.200 | 0.200 | 0.100 |
| chb23 | 0.9524 | 1.0000 | **0.778** | **2.106** | **2.221** |

chb04 -- `k5only`'s worst patient -- gains 25pp. But chb23's false-alarm rate
triples, landing within 5 % of the teacher's own 2.221. The student converged
onto the teacher's FAILURE MODE more completely than onto its successes, and
chb23 is the same patient that carried `k5only`'s entire apparent FAR advantage
in Phase 2 (section 15.2 of the reality check, where that advantage was shown to
be one patient and not significant).

### Provenance

Rows 41-42: `run_phase5_l3.sh` at commit 705fad3 (montage fix), arm `L3`.
The `L3single` control arm was still running when these were read; without it,
"more channels" and "more teacher capacity" are not yet separated for L3 --
though L8 already supplies a same-input, higher-capacity teacher and points the
same way.
Row 43: `run_phase6_l8.sh` at commit abd0511, `alpha=0.5`, `WORKERS=7`,
teacher = row 33 checkpoints, paired by seed.


## 2h. Row 44 -- narrowing the context block is NOT free (09-02)

Rung 1 of the capacity ladder: `context_channels` 64 -> 16, nothing else
changed. 367,664 MACs against the control's 585,920, a 37% cut.

| | ctx16 (row 44) | control (row 32) | delta, 95% CI |
|---|---:|---:|---|
| sens macro | 0.9126 | 0.9358 | **-2.33pp** [-6.60, +0.60] |
| sens micro | 0.9221 | 0.9351 | **-1.30pp** [-4.09, +0.58] |
| FAR/h | 0.2378 | 0.2216 | +0.0162 [-0.036, +0.069] |
| delay mean | 18.10 | 18.83 | -0.73 [-1.98, +0.28] |

Three seeds, 66 folds each, `hysteresis_widegrid`, cluster bootstrap over 13
patients. Paired sign tests: FAR p = 1.0000, sensitivity p = 0.6250.

### What this does and does not establish

Nothing reaches significance. But the sensitivity point estimate is down on
BOTH aggregations, and the interval is about 7pp wide -- this cohort cannot
resolve a 2.3pp difference either way. So the honest statement is **neither
"free" nor "costly": undetermined at this sample size**, with the point
estimate pointing down.

The prediction that motivated this rung was that the context block is mostly
wasted computation -- its second conv is a k5 at dilation 16 spanning 65 samples
of a 32-sample sequence, so most taps read padding. That reasoning is sound
about the ARITHMETIC and is not supported as a claim about ACCURACY: whatever
those taps compute, removing three quarters of the block's width did not come
for nothing.

### One number in the output that must not be reported as a finding

`worst_patient_far_per_hour` came back "B better", CI [+0.019, +0.253],
excluding zero. That is the max over 13 clusters -- the statistic section 15.2
of the reality check already established gives degenerate intervals and is not
testable. The testable version, the paired per-patient sign test, is **p =
1.0000**. The apparent effect is one patient, chb23 (0.985 vs 0.778), which is
the same patient that carried k5only's apparent FAR advantage in Phase 2 and
the same one L8 damaged. chb23 dominates any max-based false-alarm statistic in
this cohort and should be named whenever one is quoted.

### What it means for rung 2

`k5only_wide` is built on ctx=16, so it starts about 2.3pp down and has to
recover that before any widening shows up as a gain against the control. This
makes `wide_vs_ctx16` -- which isolates the stage width -- the comparison to
read first, and `wide_vs_control` meaningless without it.

If `wide` lands below the control, "wider stages do not help" is NOT the only
reading available; "wider stages helped but did not pay for the context" fits
equally. The rung that would separate them is ctx=32 with stages (20,32,40,60)
at 562,912 MACs -- half the context kept, stages 25% wider, and still fewer
MACs than the control. Measured with thop, not yet run.


## 3b. The teacher montage each L3 fold actually gets (01-09)

Phase 5 aborted at fold 17 of 66: `chb04` has EDFs with either 23 or 24 channels
(5 files and 35), and the first version of `load_multichannel_for_fold` required
one channel count per fold. Refusing was right about padding -- it would have
fed the teacher fabricated signal -- but wrong as a policy, since it would have
dropped 49 folds. The rule is now an intersection of channel NAMES, sorted, so
row order is identical across files instead of inherited from EDF ordering.

`scripts/check_fold_montage.py profile=server data=chbmit`, header reads only:

| subject | per-file channel counts | common channels |
|---|---|---:|
| chb01–chb11, chb22, chb23 | `{23: n}` | **22** |
| chb04 | `{23: 5, 24: 35}` | **22** |
| chb15 | `{26: 1, 32: 37}` | **26** |
| chb17 | `{23: 18, 18: 1}` | **18** |

`66/66 folds usable; common channels min=18 median=22 max=26`.

Three things to carry into the write-up:

1. **23 signals but 22 names.** CHB-MIT repeats a label (`T8-P8`) on two rows.
   Intersecting by name collapses the duplicate and `load_edf_multichannel`
   resolves it to its first occurrence -- deterministic, and the duplicated row
   carried no information anyway.
2. **Teacher capacity is not uniform across patients** (18 / 22 / 26 channels).
   Harmless mechanically, since each fold trains its own teacher, but it is a
   confound to declare: the L3 arm's teacher is stronger for chb15 than chb17.
   The L3single control uses the same per-fold architecture on one channel, so
   the L3-vs-L3single contrast stays clean WITHIN a patient.
3. **chb17 is pulled down to 18 by a single file.** chb17 is already the weakest
   patient (3 seizures) and holds the worst-patient gate. If L3 fails to improve
   chb17 specifically, its thinner teacher is one explanation to rule out before
   concluding anything about distillation.


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
