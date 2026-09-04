# WearSeizure-1D

Single-channel EEG seizure **detection** under a leakage-safe protocol, sized for a streaming
INT8 1D-CNN accelerator. Targets IEEE TBioCAS.

**Dataset**: CHB-MIT, 13 single-channel-eligible cases, 77 seizures, 599.5 recording hours.
**Evaluation**: `patient_specific_loso_edf`, 66 folds, **185.0 h** of continuous test exposure,
3 seeds. Thresholds frozen on validation; splits taken by recording before any filtering,
normalisation or windowing.

Software pipeline only (gates G0-G2). RTL is deferred — the interface contract is in
`docs/RTL_INTERFACE_SPEC.md`, and the frozen model's per-layer spec is in
`docs/MODEL_CARD_k5only.md`.

---

## Current result

`wearseizure1d_k5only`, cohort pre-training + distillation (L1 + L8), 3 seeds x 66 folds:

| | |
|---|---:|
| event sensitivity (macro / micro) | **0.9489 / 0.9567** |
| false alarms per hour | 0.2937 |
| mean detection delay | 17.75 s |
| worst-patient sensitivity (>=5 seizures) | 0.9333 |
| parameters / MACs | **11 786 / 585 920** |
| on-chip SRAM, INT8 | **18.2 KiB** |

Architecture is frozen; the weights are not. `L1 + L4` trades 1.1pp of sensitivity for a much
better false-alarm rate (0.1904/h, worst-patient 0.5701 vs 2.1065). Swapping recipes costs no
RTL — same layers, same MACs, same buffers.

---

## Benchmarks

### A. Published numbers, each under its own protocol

These are **not** directly comparable to each other or to us — protocols, channel counts and
test exposure all differ. The table exists to show what the literature reports, and section C
explains the gap.

| Work | Ch | Protocol | Sens | FAR/h | Delay | Params | MACs |
|---|---|---|--:|--:|--:|--:|--:|
| Chung 2024, *Front. Neurol.* 15:1389731 | **1** | patient-specific; **1-sample** stride; thresholds tuned on eval data; ~91 h | 99.62 % | 0.22 | 3.3 s | — | — |
| Ultra-light 3D-CNN, *BSPC* (2025) | multi | patient-specific | 99.24 % | 0.53 | 4.97 s | **6 540** | 2 390 000 |
| Multi-scale channel attention (2023) | multi | segment-level | 98.3 % | — | — | 88 000 | 2 680 000 |
| Ali 2024, *R. Soc. Open Sci.* 11:230601 | 18 | **cross-subject**, full corpus | 75.34 % | 4.79 | — | — | — |
| **WearSeizure-1D (this work)** | **1** | **leakage-safe, 185 h, thresholds frozen on val** | **94.89 %** | 0.29 | 17.75 s | 11 786 | **585 920** |

Two things this table does *not* license:

- **Not the smallest model.** 11 786 parameters is 1.8x the 3D-CNN's 6 540. The computational
  claim is about **MACs**, not parameter count.
- **Part of the MAC advantage is the single channel**, not the architecture. That is the
  contribution, but it should be said rather than hidden.

### B. Same protocol, same code, same data — the comparison that is valid

Every row: 3 seeds, 66 folds, 185.0 h, cohort pre-training, thresholds frozen on validation.

| Architecture | Sens macro | FAR/h | Delay | Worst-pt sens | Params | MACs |
|---|--:|--:|--:|--:|--:|--:|
| `baseline_frontiers2d` (Chung 2024 architecture) | **0.9726** | 0.2922 | 17.22 | 0.9500 | 4 546 | 2 523 328 |
| `baseline_compact1d_7k` | 0.9440 | 0.3353 | 16.97 | 0.9667 | 7 570 | 1 398 928 |
| `wearseizure1d_k5only` | 0.9358 | **0.2261** | 18.83 | 0.8714 | 11 786 | **585 920** |
| **`wearseizure1d_k5only` + L8** | **0.9489** | 0.2937 | 17.75 | 0.9333 | 11 786 | **585 920** |

The Chung architecture reproduced honestly reaches **0.9726**, not 0.9962. On this cohort the
architectures are **statistically indistinguishable** on sensitivity, FAR and delay (paired
cluster bootstrap, 13 patients), so the remaining axis is cost — and `k5only` holds the operating
point at **4.3x fewer MACs**.

### C. Where 99.62 % comes from — the protocol ladder

One body of code, one body of data, 66 folds. Only the partitioning rule changes.

| Rung | Split | Normalised on | Threshold on | Window sens | Accuracy | Test/train overlap |
|---|---|---|---|--:|--:|--:|
| **A** as published | **random windows** | everything | test | **0.9229** | 0.9968 | **99.6 %** |
| **B** | by recording | everything | test | 0.6173 | 0.9887 | 0.0 % |
| **C** | by recording | train only | val | **0.6033** | 0.9888 | 0.0 % |

(`wearseizure1d_k5only`; `baseline_frontiers2d` and `baseline_compact1d_7k` behave the same way —
see `docs/EXPERIMENT_LOG_G1a.md` section 2j.)

**Splitting windows at random inflates sensitivity by 31 points.** At a 4 s window and 1 s stride,
adjacent windows share 75 % of their samples, so 99.6 % of test windows have a near-duplicate in
training — measured, not asserted.

**And accuracy cannot see any of it.** Across all seven measured cells accuracy spans **1.15pp**
while sensitivity spans **32pp**. At 0.62 % ictal prevalence a model that never predicts a seizure
scores **99.38 %**, so the best cell in the table sits 0.30pp above predicting nothing. Accuracy on
this data has almost no dynamic range, which is why it is never reported here without prevalence
beside it.

Rung A is a **partial** reproduction: it kept this project's 1 s stride, while the published
protocol slides by **one sample**. `scripts/run_stride_sweep.sh` tests whether that accounts for
the rest of the gap to 99.62 %.

---

## Quickstart (local, synthetic)

This machine needs no CHB-MIT data and no GPU. A synthetic generator produces the same manifest
schema as the real dataset, so the whole pipeline runs on CPU in minutes.

Use a native Windows Python (python.org / Store / `py`), not an MSYS2/MinGW `python` — PyPI
wheels for torch/scipy will not install on that ABI.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r environment/requirements-local.txt   # torch CPU index has no scipy/numpy
pip install -e ".[dev,analysis]"

python scripts/generate_synthetic_dataset.py
python scripts/make_manifest.py profile=local_synthetic
python scripts/make_splits.py  profile=local_synthetic
python scripts/train.py        profile=local_synthetic
python scripts/evaluate.py     profile=local_synthetic

pytest tests/unit -v
pytest tests/integration -v -m integration
```

## Running on a server

Set both variables, then always pass **`profile=server data=chbmit`** — the `profile` group picks
device and paths, but `data` still defaults to `synthetic`, and `profile=server` alone would point
the synthetic loader at clinical recordings. `utils/profile_guard.py` refuses the combination.

```bash
export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0
export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts

python scripts/make_manifest.py profile=server data=chbmit
python scripts/make_splits.py   profile=server data=chbmit
python scripts/train.py         profile=server data=chbmit train.pretrain.enabled=true
python scripts/evaluate.py      profile=server data=chbmit
```

`scripts/server_bootstrap.sh` has the full first-time setup. `docs/SERVER_INVENTORY.md` covers the
four-server inventory; SERVER-02/03/04 share `~/Manh` over NFS, so **the checkout is one directory,
not three** — run `scripts/check_servers.py` before starting work on a second host.

**Discipline**: the server only ever checks out a reviewed commit SHA from `main`, never a branch
tip. Runs touch clinical data and feed a publication.

Long runs go through a script in `scripts/run_phase*.sh`, never a pasted command block. Every
hand-typed block in this project's history shipped a defect (missing `ulimit -n`, a dropped `&`,
an ignored `TAG`); every scripted one did not.

## Useful overrides

| Override | What it does |
|---|---|
| `train.seeds=[0,1,2]` | Lever L7. One run per seed into its own `seed<N>/`; `evaluate.py` reports mean ± std. Without it no comparison has an error bar, and the top configurations are ~1.4pp apart — about one seizure in 77. |
| `train.pretrain.enabled=true` | Lever L1. Cohort pre-training then per-patient fine-tuning. The largest gain on record, +6 to +9pp. |
| `train.distill.teacher_model=baseline_frontiers2d` | Lever L8. Distil a finished single-channel run into the small student. |
| `train.model_selection=val_auprc` | Lever L4. Select checkpoints on AUPRC instead of cross-entropy — better FAR, same sensitivity. |
| `eval.gates_path=configs/eval/gates_v2_proposed.yaml` | Score against the v2 gate table instead of v1. |

Compare two runs with `scripts/paired_bootstrap.py A_DIR B_DIR --all-metrics` — patient-clustered
paired bootstrap from saved `*.metrics.json`, no GPU and no retraining. Use it instead of comparing
point estimates.

## Tooling

| Script | Answers |
|---|---|
| `check_servers.py` | Which host is running what, and does anything collide? |
| `estimate_remaining.py` | How much longer, per host, measured from artifact timestamps |
| `check_phase_complete.sh` | Did a phase finish, or abort? They look identical from outside |
| `hardware_spec.py` | Per-layer shapes, MACs, line buffers and SRAM footprint |
| `check_fold_montage.py` | Which EEG channels each fold's teacher would get |
| `summarise_leaky_repro.py` | The protocol-ladder table above |

## Layout

```
configs/         Hydra groups (profile, data, split, model, window, postprocess, precision, eval)
docs/            Protocol, gates, experiment log, model card, hardware handoff, RTL interface
scripts/         CLI entry points and the phase runners
src/wearseizure  data, signal, models, quant, postprocess, eval, training, rtl_interface
tests/           Unit (fast, synthetic) and integration smoke tests
```

Start with `docs/EXPERIMENT_LOG_G1a.md` — every real-data run, in order, with what it disproved.
`docs/PROTOCOL.md` defines the leakage-safe protocol; `docs/RESEARCH_REALITY_CHECK.md` records
which of this project's own earlier claims have since been overturned.
