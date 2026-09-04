# Running WearSeizure-1D

Everything about installing, running and reproducing. The README covers what the
project found; this covers how to make it run.

---

## Local, on synthetic data

No CHB-MIT data and no GPU needed. A synthetic generator produces the same
manifest schema as the real dataset, so the whole pipeline — manifest → split →
window → train → evaluate — runs on CPU in minutes.

Use a native Windows Python (python.org / Store / `py`), **not** an MSYS2/MinGW
`python` that may come first on PATH in Git Bash: PyPI wheels for torch and
scipy will not install on that ABI.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r environment/requirements-local.txt   # the torch CPU index has no scipy/numpy
pip install -e ".[dev,analysis]"

python scripts/generate_synthetic_dataset.py
python scripts/make_manifest.py profile=local_synthetic
python scripts/make_splits.py  profile=local_synthetic
python scripts/train.py        profile=local_synthetic
python scripts/evaluate.py     profile=local_synthetic

pytest tests/unit -v
pytest tests/integration -v -m integration
```

Synthetic results are meaningless clinically — `enforce_gates=false` for that
profile. The point is that every code path executes.

---

## On a server, on CHB-MIT

### Two variables, and both config groups

```bash
export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0
export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts
```

`configs/profile/server.yaml` interpolates both. They are the first thing to go
missing on a fresh host, because they normally live in `~/.bashrc`.

Always pass **`profile=server data=chbmit`** — both. The `profile` group picks
the device and paths, but `data` still defaults to `synthetic`, so
`profile=server` alone points the synthetic loader at clinical recordings.
`utils/profile_guard.py` refuses that combination outright.

```bash
python scripts/make_manifest.py profile=server data=chbmit
python scripts/make_splits.py   profile=server data=chbmit
python scripts/train.py         profile=server data=chbmit train.pretrain.enabled=true
python scripts/evaluate.py      profile=server data=chbmit
```

`scripts/server_bootstrap.sh` has the full first-time setup (conda env, clone,
first manifest validation). Read it before running — it is deliberately not
something to pipe into a shell.

### The shared disk

SERVER-02/03/04 share `~/Manh` over NFS. **The checkout is one directory, not
three.** Consequences that have already bitten:

- `> run.out` from two hosts truncates one file. Put the hostname in the
  redirect: `> "run_$(hostname).out"`.
- `git checkout` on one host changes the code under the others. Their running
  Python is safe (modules are already loaded), but the *next* job a phase script
  launches picks up the new code.
- Two hosts training the same `(model, seed)` write the same cohort-init `.pt`
  files and can read each other's partial writes. That does not raise — it
  produces a wrong result silently.

Run `python scripts/check_servers.py` before starting work on a second host. It
reads the run directories and reports collisions.

### Discipline

The server only ever checks out a **reviewed commit SHA** from `main`, never a
branch tip: runs touch clinical data and feed a publication.

Long runs go through a script in `scripts/run_phase*.sh`, never a pasted command
block. Every hand-typed block in this project's history shipped a defect — a
missing `ulimit -n 65536` that killed the first L4 attempt at the fourth cohort
init of every combination, a dropped `&` that ran a job in the foreground until
the session closed, an ignored `TAG` that overwrote a control arm. Every
scripted one carried its guards.

### Hosts that cannot run `torch_shm_manager`

Some hosts cannot create the socket directory that PyTorch's `file_system`
sharing strategy needs, and every job dies with *"could not generate a random
directory for manager socket"*. SERVER-04 is one.

`train.py` only enables that strategy when `num_workers > 0`, so pass
`profile.num_workers=0` there. It costs nothing: this workload is bound by CUDA
kernel-launch overhead, not by data loading — the training process sits under
100 % CPU on 28 threads and the workers are idle.

For the same reason, **more workers never help and concurrent processes do**. To
go faster: raise `train.batch_size`, and run independent jobs at the same time.

---

## Levers

Each is off by default. Every number in `docs/EXPERIMENT_LOG_G1a.md` says which
were on.

| Override | Lever | Effect |
|---|---|---|
| `train.seeds=[0,1,2]` | L7 | One run per seed into its own `seed<N>/`; `evaluate.py` reports mean ± std. Without it no comparison has an error bar, and the top configurations are ~1.4pp apart — about one seizure in 77. |
| `train.pretrain.enabled=true` | L1 | Cohort pre-training then per-patient fine-tuning. **The largest gain on record, +6 to +9pp.** |
| `train.distill.teacher_model=baseline_frontiers2d` | L8 | Distil a finished single-channel run into the small student. +1.3pp macro, +2.2pp micro. |
| `train.model_selection=val_auprc` | L4 | Select checkpoints on AUPRC instead of cross-entropy. Same sensitivity, 16 % lower FAR. |
| `train.distill.enabled=true` | L3 | Multi-channel teacher. **Measured worse** — see the experiment log section 2g. |
| `train.pretrain.use_wider_corpus=true` | L5 | Pre-train on CHB-MIT cases the protocol does not evaluate. **Measured null or worse.** |
| `eval.gates_path=configs/eval/gates_v2_proposed.yaml` | — | Score against the v2 gate table instead of v1. |

Two levers are recorded as negative results deliberately. Re-running them is
waste; the reasoning is in the log.

---

## Comparing two runs

```bash
python scripts/paired_bootstrap.py A_DIR B_DIR --all-metrics
```

Patient-clustered paired bootstrap from saved `*.metrics.json` — no GPU, no
retraining. **Use it instead of comparing point estimates.** On 13 patients this
cohort cannot resolve a difference under about 3pp, and several apparent wins in
this project's history disappeared under it.

One trap it reports and you should not quote: `worst_patient_*` is a max over 13
clusters, so its bootstrap interval is degenerate. The script flags it
`DEGENERATE` and prints a paired per-patient sign test, which is the testable
version.

---

## Diagnostics

| Script | Answers |
|---|---|
| `check_servers.py` | Which host is running what, and does anything collide? |
| `estimate_remaining.py` | How much longer, per host, measured from artifact timestamps |
| `check_phase_complete.sh` | Did a phase finish, or abort? They look identical from outside |
| `hardware_spec.py` | Per-layer shapes, MACs, line buffers, SRAM footprint |
| `check_fold_montage.py` | Which EEG channels each fold's teacher would get |
| `summarise_leaky_repro.py` | The protocol-ladder table |
| `pull_results.sh` | Fetch `*.json` summaries locally; excludes `.pt` and `.npz` |

---

## What not to commit

Checkpoints, raw `.edf`, generated splits — see `.gitignore`. No absolute
Windows or server path ever goes in a config file; override via `.env` at the
repo root (see `.env.example`) or the environment. An already-exported variable
always wins.
