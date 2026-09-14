# Context brief for the paper-writing agent

Paste this whole file to the agent that drafts the manuscript in Overleaf. It is
written to be self-contained: an agent that has never seen this repository
should be able to draft from it without inventing anything.

**The single most important instruction: do not invent, round, or extrapolate a
number.** Every figure below is either measured and marked so, or explicitly
marked as unverified/not run. If the draft needs a number that is not here, ask
for it rather than producing a plausible one. Section 7 lists claims that must
never appear.

---

## 1. What the paper is about

A single-channel EEG seizure **detector** designed to run on a programmable
1D-CNN accelerator on an AMD-Xilinx Kria KV260 SoC, for a wearable device with
one electrode pair.

Two claims carry the paper:

1. **A detector constrained to one channel and ~586k MACs is competitive with
   published work at the event level**, under an evaluation protocol that is
   materially stricter than the published ones it is compared against.
2. **Most of the apparent performance of published single-channel detectors on
   CHB-MIT is protocol, not model.** This is measured, not asserted, and it is
   the paper's strongest and most defensible contribution.

Target venue: IEEE TBioCAS. There is a companion undergraduate thesis on the
RTL side; the accelerator deliverables have already been handed over and merged.

---

## 2. Dataset and protocol (state this precisely; reviewers will check)

* **CHB-MIT**, 13 of 24 cases. The restriction comes from Chung et al. 2024,
  who clinically confirmed that seizure onset is observable from one specific
  wearable electrode position in exactly those cases. Every CHB-MIT montage
  carries all four wearable positions, so this is a confirmation restriction,
  not an availability one.
* **One channel per patient**: `P7-O1` (chb02, 05, 10, 11, 15), `Fp1-F3`
  (chb03, 07, 08, 22, 23), `P8-O2` (chb01, 04, 17).
* **Window** 4.0 s at 256 Hz = 1024 samples, stride 1 s.
* **Split**: `patient_specific_loso_edf`. For each patient, each seizure-bearing
  EDF becomes one fold's test set, together with one randomly chosen
  seizure-free EDF so that false alarms are measured over ordinary time and not
  only over recordings that contain seizures. The remaining EDFs of that patient
  are split 80/20 train/val **at file level**. This yields **66 folds**,
  **77 seizures**, **185.0 h** of continuous test exposure.
* **Two-stage training**: cohort pre-training on the other 12 patients (each
  still one channel), then fine-tuning on the target patient. The target patient
  is excluded from pre-training, by identity rather than by id string — chb21 is
  chb01 recorded 1.5 years later and is excluded with it.
* **Leakage discipline**, all four points worth stating explicitly:
  split before any signal processing; causal `lfilter` bandpass 1–30 Hz with
  state reset per recording (never `filtfilt`); affine normaliser fitted on
  train only; post-processing thresholds selected on validation and frozen
  before test is touched.
* The model is **patient-specific**: 66 folds means 66 weight sets, not one
  shared model. A deployed device loads that patient's 18.2 KiB. This must be
  stated wherever the paper is compared against patient-independent work.

---

## 3. The model

| | |
|---|---|
| Architecture | `wearseizure1d_k5only`: Conv(k7,s2) stem → depthwise-separable block → three multi-scale dilated blocks (k5 branch only, dilation 1/2/4) → two dilated depthwise-separable context blocks (dilation 8/16) → GAP → FC(2) |
| Parameters | **11,786** |
| MACs per inference | **585,920** |
| Weight memory, DFP8 | **18.2 KiB** |
| Input | 1 channel × 1024 samples |
| Layers as executed by hardware | 13 conv instructions + GAP and FC on the host ARM |

---

## 4. Results that are measured and may be stated

### 4.1 Headline detection performance

**Event sensitivity 0.9495, FAR 0.2500/h**, FP32, 66 folds × 3 seeds, 185.0 h,
77 seizures.

> **Caveat that must be respected.** A second computation over the same folds at
> **seed 0 only** gives 0.9848. The 3.5-point gap between one seed and the
> three-seed mean is not yet explained. **Use 0.9495. Do not use 0.9848
> anywhere**, and do not present a range spanning the two.

### 4.2 Quantisation (DFP8) — measured on the real integer datapath

Not a fake-quantised simulation: 48-bit accumulator, requantisation by
arithmetic shift with round-to-nearest ties-away-from-zero, saturation, and the
extra quantisation points the hardware introduces between each depthwise and
pointwise pair.

| | event sensitivity | FAR/h |
|---|--:|--:|
| FP32 | 0.9848 | 0.2458 |
| DFP8 | 0.9848 | 0.3086 |
| delta | **0.00 pp**, CI [0.00, 0.00] | **+0.0628** |

(These two rows are seed 0, matched fold-for-fold — that is why they read 0.9848
rather than 0.9495. Quote the **delta**, not the absolute values, from this
table.)

Sensitivity is identical on **all 66 folds**; FAR differs on 13 of them. That
combination is what makes the zero credible rather than suspicious: the scores
moved, but no window moved far enough to flip a whole seizure event.

Two further measured results, both negative, both worth one sentence each:

* **Re-fitting thresholds on DFP8's own validation scores is worse**, not
  better: −4.55 pp sensitivity for 0.015/h of FAR. The search moved on 66/66
  folds, so this is not "nothing better was found". Cause: under LOSO most folds
  hold a single validation seizure, so selecting an operating point on
  validation overfits. This is a property of the cohort.
* **Reading the 48-bit accumulator instead of the requantised logits changes
  nothing** (0.00 pp, 0.0013/h), despite collapsing 254 distinguishable scores
  into 34, because post-processing integrates the score across windows.

### 4.3 The protocol ladder — the paper's Figure 1

Same architecture, same data, 66 folds per cell, only the split changes.

| rung | split | window sens | accuracy | test/train near-duplicate |
|---|---|--:|--:|--:|
| A | random over windows (as commonly published) | **0.9229** | 0.9968 | **99.6 %** |
| B | by recording | **0.6173** | 0.9887 | 0 % |
| C | B + no fitting leak | **0.6033** | 0.9888 | 0 % |

**Sensitivity falls 31 points; accuracy moves 0.8.** Across all seven measured
cells accuracy spans 1.15 pp while window sensitivity spans 32 pp.

The reason: at 0.62 % ictal prevalence, a model that never predicts a seizure
scores **99.38 % accuracy**. The best cell, 99.68 %, is 0.30 points above
predicting nothing. **Accuracy has no dynamic range on this data**, and any
comparison of detectors argued on accuracy is arguing on a quantity that cannot
distinguish a leaky protocol from a clean one.

Note for honesty: ictal prevalence differs by rung (0.62 % at A, 1.42 % at B/C)
because A's test set is a random 20 % of all windows. Accuracy is therefore not
comparable across rungs; sensitivity and specificity are. The accuracy finding
survives, because both trivial baselines (99.38 % and 98.58 %) sit below every
measured value.

Rung A is a **partial** reproduction and must be labelled as one: it reaches
0.9229, not the published 0.9962.

### 4.4 Channel ablation -- how much the one-channel constraint costs

Three arms, identical except channel count; montages taken verbatim from Chung
et al. 2024. **66 folds x 3 seeds = 198 per arm.** Trained from scratch per
fold, no cohort pre-training and no distillation, so absolute numbers sit below
§4.1 and only cross-arm comparison is valid.

| arm | params | event sens | FAR/h | segment sens | accuracy | AUROC |
|---|--:|--:|--:|--:|--:|--:|
| 1 channel | 11,786 | 0.9179 | 0.2798 | **0.4936** | 0.9896 | 0.8870 |
| 4 channels | 11,954 | 0.9331 | 0.3334 | **0.5936** | 0.9910 | 0.9237 |
| 18 channels | 12,738 | 0.9520 | 0.2625 | **0.7051** | 0.9937 | 0.9586 |

Paired bootstrap against the 18-channel arm, clustered by patient:

| | event sensitivity | segment sensitivity |
|---|--:|--:|
| 1ch vs 18ch | **-3.41 pp, CI [-5.83, -0.93]** | **-21.16 pp, CI [-26.28, -16.47]** |
| 4ch vs 18ch | -1.89 pp, CI [-5.89, +1.97] | **-11.16 pp, CI [-15.18, -7.13]** |

Four findings, all reportable:

1. **One channel is measurably worse than eighteen at event level: -3.41 pp,
   interval excluding zero, 2.62 seizures out of 77.** Report this cost
   plainly. It is what the wearable form factor buys.
2. **Post-processing absorbs 84 % of the deficit.** 21.16 pp at segment level
   becomes 3.41 pp at event level, because smoothing, hysteresis and a
   run-length filter integrate the score across many consecutive windows. This
   is the mechanism worth a subsection.
3. **The cost is concentrated in the last step.** Four channels against
   eighteen still spans zero (-1.89 pp); one channel does not. Most usable
   information survives to four electrodes and is lost going to one.
4. **The published channel penalty is an order of magnitude too small.** Chung
   et al. report 1.9 points of segment sensitivity from 18 channels to 1;
   without the leak it is 21.2. Their random split over overlapping windows
   lifts every arm toward its ceiling and compresses the distance between them.

Accuracy moves 0.41 pp across an eighteen-fold change in input channels -- a
third independent demonstration that it is the wrong metric here. AUROC, being
threshold-free, is not blind: 0.8870 -> 0.9237 -> 0.9586.

### 4.5 Training-recipe levers already measured

| lever | what it added | result |
|---|---|---|
| L1 cohort pre-training | other 12 patients, one channel each | largest improvement recorded in the project |
| L5 wider corpus | pre-training hours 553 → 2085 (×3.7) | **negative**: no gain; the one-position variant lost 4.34 pp, CI [−8.61, −0.86] |
| L3 multi-channel teacher | teacher reads 18–26 channels, student reads 1 | **negative** |
| L8 same-channel teacher | teacher reads what the student reads | +1.31 pp, CI [−2.52, +6.03], **not significant** |

The mechanism ties L3, L5 and L8 together and is worth a short subsection:
**distillation helps when the teacher's advantage is capacity, and hurts when it
is information.** A soft target is only imitable if the student could in
principle compute it; a teacher whose confidence depends on channels the student
will never see pulls the student toward a number it cannot derive.

---

## 5. Related work — verified figures only

**Chung et al. 2024**, *Frontiers in Neurology* 15:1389731,
doi:10.3389/fneur.2024.1389731. The closest comparison. Verified from the full
text:

| | segment sens | segment acc | event sens | FAR/h |
|---|--:|--:|--:|--:|
| 18 channels | 98.66 ± 1.19 % | 98.47 % | 100 % | 0.30 ± 0.47 |
| 4 channels | 97.31 ± 3.78 % | 97.73 % | 97.05 ± 9.23 % | 0.40 ± 0.77 |
| 1 channel | 96.76 ± 3.97 % | 98.18 % | 99.62 ± 1.39 % | 0.22 ± 0.34 |

Protocol: patient-specific, k-fold with k = number of seizure-bearing EDF files
per case. **Two evaluations**, and conflating them is the most likely error in
the draft:

* *segment-level*: segments pooled from k−1 seizure files plus all seizure-free
  files, split 7:2:1 **at random over overlapping windows** — this is the leaky
  stage, and the stage rung A reproduces;
* *event-level*: the held-out k-th file, sliding window with 1 s step — this is
  where 99.62 % and 0.22/h come from, and it is **not** a leaky evaluation.

**Be fair to them.** Do not describe their evaluation as simply "leaky". Their
headline number is event-level on a held-out recording. The honest statement of
the difference is narrower and still favourable: this work additionally holds
out a seizure-free recording in every test set, evaluates over 185 h against
their ~91 h, and freezes post-processing thresholds on validation before test.

**Busia et al. 2025**, *IEEE TBioCAS* 19(6):1175–1186,
doi:10.1109/TBCAS.2025.3575327. Verified from the PDF: **4 channels** (F7-T7,
T7-P7, F8-T8, T8-P8), leave-one-record-out, and *"for the training and test of
the models, we consider only records including a seizure event"* — so their
0.30/h false alarm rate is measured only over seizure-bearing recordings.
8 subjects, 43–44 seizures, 61 h.

**Zhu et al. 2021**, ASICON, doi:10.1109/ASICON52560.2021.9620467. 7,010
parameters, ~6.32 M MACs, 16-bit, 97.35 % accuracy claimed on CHB-MIT.
**Channel count and evaluation protocol could not be verified** — IEEE access
was blocked. Either obtain the full text or mark the row as unverified. **Do not
state that it is single-channel.**

**Ali et al. 2024**: 75.34 % at 4.79 FAR/h with **18 channels**, zero-shot /
patient-independent. Only compare against this with the protocol difference
stated in the same sentence.

---

## 6. Hardware side (one section, already delivered)

Deliverables merged into the RTL team's repository: folded and quantised DFP8
weights, a bit-exact NumPy golden model, per-layer test vectors, and a manifest
with calibrated per-layer shifts.

Measured, quotable:

* `output_shift` ranges 5–10 across layers and no two adjacent layers agree;
  the flat placeholder value it replaced could not be right for any network.
* Weight SQNR 34–47 dB; **activation SQNR only 22–31 dB**. The activations are
  the quantisation bottleneck, so per-channel *weight* scales would attack the
  wrong term — measured at +6.9 dB on the best layer and still not worth it.
* Peak accumulator on real data: **19,023 — 15 bits of the 48 provided**.
* The reference RTL had no activation unit and no dilation support. The network
  needs ReLU after 10 of 15 layers, non-uniformly, and dilation 2/4/8/16 in four
  depthwise layers. The RTL team has since added a RELU_EN instruction bit and
  widened the padding field.

---

## 7. Claims that must NOT appear in the draft

* **98.48 %** as a headline sensitivity. It is one seed and the discrepancy with
  the three-seed 94.95 % is unexplained.
* **"One channel is equivalent to 18 channels."** The event-level interval
  spans zero but reaches −10.53 pp on one seed. Under this project's own rule —
  rank by the worst case the interval allows, not by the point estimate — this
  is not established.
* **"Adding channels does not help."** Measured false: 19.7 points of segment
  sensitivity and 8.5 points of AUROC.
* **"Chung et al. evaluated with data leakage."** True of their segment-level
  stage only; their headline is event-level on a held-out recording.
* **Zhu et al. described as single-channel.** Unverified.
* **Any zero-shot / patient-independent number.** That branch is implemented but
  has never been run.
* **Detection-delay gate compliance.** With a 4 s window and causal `window_end`
  alarm timestamping the floor is 4.0 s; the best model reaction measured is
  4.06 s. Delay may be reported with that decomposition but no gate is met.
* **Any accuracy-based superiority claim.** The paper's own evidence is that
  accuracy cannot distinguish protocols on this data.
* **Worst-patient and delay figures** — not verified in the current run set. Ask
  before using.

---

## 8. Where the numbers live

* `docs/EXPERIMENT_LOG_G1a.md` — every run, by row number. §2e (L5), §2j
  (protocol ladder), §2m (numeric format), §2n (channel ablation).
* `docs/RESEARCH_REALITY_CHECK.md` — protocol analysis and gate arithmetic.
* `docs/MODEL_CARD_k5only.md` — the frozen model.
* `docs/GIAI_TRINH_DU_LIEU.md` — the data-and-channels argument in Vietnamese.
* `docs/comparison_table.pdf` — an earlier comparison table. **It predates the
  channel ablation and contains at least two errors now known**: Busia et al.
  listed as 1 channel (it is 4), and Zhu et al.'s channel count unverified.
  Rebuild rather than reuse.

## 9. Still running

Nothing. The channel ablation completed at three seeds and §4.4 is settled --
draft it as written there, including the 3.41-point cost of the single-channel
constraint.

Open items that are *not* blocking and must simply not be claimed: the
zero-shot / patient-independent branch has never been run, worst-patient and
detection-delay figures are not verified in the current run set, and the
94.95 % / 98.48 % discrepancy in §4.1 is unexplained.
