# WearSeizure-1D

Clinically guided single-channel EEG seizure **detection** with a mixed-precision streaming
1D-CNN accelerator. Software pipeline for the IEEE TBioCAS-targeted research program defined
in the project Research Decision Memo (dataset: CHB-MIT, 13 single-channel-eligible cases,
77 seizures, 599.5 recording hours).

This repository currently covers the **software pipeline only** (Gates G0–G2): leakage-safe
data protocol, baseline reproduction, the WearSeizure-1D model, and QAT/integer-reference
quantization. The training server is now confirmed (see "Training server" below); the FPGA
board is not yet, so RTL work (G3+) stays deferred — see `docs/RTL_INTERFACE_SPEC.md` for the
interface contract prepared in advance.

## Why local vs. server

This machine has **no CHB-MIT data and no GPU**. All code is developed and tested here against
a **synthetic EEG generator** (`src/wearseizure/data/synthetic.py`) that produces data with the
same manifest schema as the real dataset, so the entire pipeline — manifest → split → windowing →
train → QAT → evaluate → report — can be exercised on CPU in minutes. The training server
(SERVER-02, see "Training server" below) already has real CHB-MIT data; it runs the exact
same code, switched to `profile=server`.

## Quickstart (local, synthetic)

Use a native Windows Python (python.org / Microsoft Store / py launcher), not
an MSYS2/MinGW `python` that may be first on PATH in a Git Bash shell -- PyPI
wheels for torch/scipy/numpy won't install on that ABI.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# Two separate installs: the PyTorch CPU index does not host scipy/numpy/etc.
pip install -r environment/requirements-local.txt
pip install -e ".[dev,analysis]"

python scripts/generate_synthetic_dataset.py
python scripts/make_manifest.py profile=local_synthetic
python scripts/make_splits.py profile=local_synthetic
python scripts/train.py profile=local_synthetic
python scripts/evaluate.py profile=local_synthetic

pytest tests/unit -v
pytest tests/integration -v -m integration
```

## Config profiles (Hydra)

- `profile=local_synthetic` — `device=cpu`, `data_root=./data/synthetic/generated`,
  `enforce_gates=false` (synthetic data has no clinical meaning, so KPI gates from Table 6 of
  the memo are not enforced).
- `profile=server` — `device=cuda`, `data_root=${oc.env:CHBMIT_RAW_DIR}`, `enforce_gates=true`.

No absolute Windows or server path is ever committed to a config file — override paths via `.env`
at the repo root (see `.env.example`) or the `WEARSEIZURE_ARTIFACTS_DIR` / `CHBMIT_RAW_DIR`
environment variables. `.env` is read at import time by `utils/env.py`; an already-exported
variable always wins, and `~` is expanded. If a `profile=server` run has neither, it now stops
with a one-line message instead of a Hydra interpolation traceback — the failure happens while
resolving `hydra.run.dir`, before any script code runs, so it cannot be caught any later.

## Training server

**SERVER-02** is the primary training target — see `docs/SERVER_INVENTORY.md` for the full
4-server inventory and why. Quick facts:

- Dataset already in place: `CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0` (standard PhysioNet
  layout, no preprocessing needed — `data/chbmit_summary_parser.py` reads it directly).
- Suggested artifacts path: `WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts`.
- GPU: Quadro RTX 8000, 48GB VRAM — far more than this ~14k-parameter model needs; GPU class was
  not the deciding factor, data locality was (`~/Manh` sits on an NFS share visible to all four
  servers, so compute can move to SERVER-03/04 later without copying the dataset).
- FPGA board (expected PYNQ-Z2/Zynq-7020) is still unconfirmed — RTL work (Gate G3+) stays
  deferred regardless of server readiness; see `docs/RTL_INTERFACE_SPEC.md`.

Full step-by-step bootstrap (clone, conda env, env vars, and the first real-data manifest/split
validation) is in `scripts/server_bootstrap.sh` — read it before running, it also documents a
review-then-delete step for cleaning up old experiment directories and is intentionally not
piped straight into a shell.

Checklist:

1. Create a dedicated conda env on the server (`scripts/server_bootstrap.sh` has the exact
   commands) — the shared base env is not assumed to already have everything this project needs.
2. Set `CHBMIT_RAW_DIR` and `WEARSEIZURE_ARTIFACTS_DIR` (see values above) in the server's `.env`
   at the repo root, or `export` them in the shell. A `.env` outside the repo root is not read.
3. **Discipline**: the server only ever checks out a reviewed commit SHA from `main` —
   never a floating branch tip — because runs touch clinical data and feed a publication.
   `git fetch && git checkout <sha>` before every run.
4. Run, in order: `make_manifest.py` → `make_splits.py` → `train.py` → `evaluate.py`,
   all with **`profile=server data=chbmit`**. Both are required: the `profile` group
   selects the device and paths, but `configs/config.yaml` defaults to
   `data: synthetic` and no profile changes it — `profile=server` alone would point
   the synthetic loader at the clinical recordings and look for a
   `synthetic_manifest.csv` that isn't there. `scripts/server_bootstrap.sh` has always
   passed both; this checklist previously said only `profile=server`, which is how the
   mistake gets made. `utils/profile_guard.py` now refuses the combination outright.
   To enable cohort pre-training (lever L1), add `train.pretrain.enabled=true` — see
   `scripts/pretrain_cohort.py` to build the initialisations in parallel first.
5. Pull result summaries back with `scripts/pull_results.sh` for local analysis in `notebooks/`.
6. Do not commit checkpoints, raw `.edf` files, or generated splits — see `.gitignore`.

**One-time migration.** Per-fold artifacts now live one level deeper, under
`<model>/<split>/<window>/seed<N>/`, so that multi-seed runs cannot overwrite each other.
Existing checkpoints from before this change sit in the `<window>` directory; adopt them as seed 0
rather than re-training 66 folds:

```bash
cd "$WEARSEIZURE_ARTIFACTS_DIR/wearseizure1d/patient_specific_loso_edf/w4s_stride1s"
mkdir -p seed0 && mv ./*.pt ./*.json seed0/
```

`train.py` / `evaluate.py` print this exact command if they find artifacts in the old location.

## Runtime switches worth knowing

| Override | What it does |
|---|---|
| `train.seeds=[0,1,2]` | Lever L7. Trains every fold once per seed into its own `seed<N>/`; `evaluate.py` then reports mean ± std and gates the **mean**. Default `null` = one run at `cfg.seed`, exactly as every number in `docs/EXPERIMENT_LOG_G1a.md` was produced. |
| `train.pretrain.enabled=true` | Lever L1. Cohort pre-training then per-patient fine-tuning — the change that produced the largest jump on record (0.8806 → 0.9218). |
| `train.pretrain.use_wider_corpus=true` | Lever L5. Also pre-trains on the CHB-MIT cases the protocol does *not* evaluate on. Evaluation stays at 13 cases / 66 folds / 185.0h; only the pre-training pool grows. Requires `make_manifest.py` to have been re-run. |
| `eval.gates_path=configs/eval/gates_v2_proposed.yaml` | Score against the proposed v2 gate table instead of v1. |
| `postprocess.objective=min_delay` | Spend surplus FAR budget on detection speed instead of sensitivity. |

`scripts/paired_bootstrap.py A_DIR B_DIR --all-metrics` compares two runs with a patient-clustered
paired bootstrap, from saved `*.metrics.json` only — no GPU, no re-training. Use it instead of
comparing point estimates: the three best configurations to date are ~1.4pp apart on 77 seizures,
which is about one seizure.

## Repository layout

```
configs/        Hydra config groups (profile, data, split, model, window, postprocess, precision, eval)
data/           Manifest schema + small synthetic example manifest (no real data)
docs/           Protocol, architecture, gates, and RTL interface spec
notebooks/      Analysis notebooks (data audit, baseline repro, ablation, QAT)
scripts/        CLI entry points (make_manifest, make_splits, train, evaluate, export, sweep)
src/wearseizure Library code (data, signal, models, quant, postprocess, eval, training, rtl_interface)
tests/          Unit tests (fast, synthetic) and integration smoke tests
```

See `docs/PROTOCOL.md` for the leakage-safe data protocol and `docs/GATES.md` for the KPI gates
this pipeline is designed to satisfy once run against real data on the server.
