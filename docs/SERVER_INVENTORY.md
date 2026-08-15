# Training server inventory

Collected 2026-08. Source of truth for `num_workers`, `OMP_NUM_THREADS` /
`MKL_NUM_THREADS`, and which machine to target with `profile=server`. If any
of this is re-measured later, update this file rather than letting the
numbers drift from what's actually running.

## Summary

| Host | OS | Kernel | CPU | Physical cores | Logical CPUs | RAM | GPU | VRAM | Driver |
|---|---|---|---|---:|---:|---|---|---|---|
| **SERVER-01** | Ubuntu 22.04.5 | 6.8.0-124-generic | i9-10940X @ 3.30GHz | 14 | 28 | ~251 GiB | Quadro RTX 8000 | 48 GB | 595.71.05 |
| **SERVER-02** | Ubuntu 20.04.6 | 5.15.0-139-generic | i9-10940X @ 3.30GHz | 14 | 28 | ~188 GiB | Quadro RTX 8000 | ~48 GB | TBD (check `nvidia-smi`) |
| **SERVER-03** | Ubuntu 22.04.5 | 5.15.0-186-generic | i9-10940X @ 3.30GHz | 14 | 28 | ~251 GiB | RTX 3090 | 24 GB | 580.126.20 |
| **SERVER-04** | Ubuntu 22.04.5 | 5.15.0-186-generic | i9-10940X @ 3.30GHz | 14 | 28 | ~251 GiB | RTX 3090 | 24 GB | 580.173.02 |

All four share the same CPU model (14 physical cores / 28 threads, 1 socket).
Use 14 as the practical ceiling for DataLoader `num_workers` and CPU-thread
env vars on any single node; start at 4-8 when a GPU job is already resident
on the box, per the original inventory doc's own guidance.

## SERVER-02 -- primary training target for WearSeizure-1D

Chosen because the CHB-MIT dataset is already there (no need to move ~42GB)
and its 48GB Quadro RTX 8000 has enormous headroom for a ~14k-parameter,
~0.64M-MAC model -- GPU class is not the bottleneck here, data locality is.

- Dataset: `~/Manh/datasets/CHB-MIT/1.0.0` -- standard PhysioNet layout
  (`chbXX/chbXX-summary.txt` + `chbXX/chbXX_YY.edf` per subject), matching
  `src/wearseizure/data/chbmit_summary_parser.py` and `data/io_edf.py`
  exactly, no code changes needed.
- `~/Manh` sits on an NFS share (`192.168.50.10:/home/ubuntu`) mounted across
  all four servers, so training could later move to SERVER-03/04 (matched
  24GB pair, newer Ubuntu 22.04) without copying the dataset -- useful if
  multi-seed ablation needs to run in parallel across boxes.
- Ubuntu 20.04 (older than 01/03/04) has no practical impact here: the model
  is tiny and any reasonably modern PyTorch/CUDA wheel will run fine. Confirm
  the exact driver with `nvidia-smi` before pinning a specific `pytorch-cuda`
  version (see `scripts/server_bootstrap.sh`).
- Three old, unsuccessful experiment directories that used to clutter
  `~/Manh` (`1D-CNN-Accelerator-for-EEG_Detection`,
  `-q1`, `-main`) have been removed (2026-08).
  Conda env `chbmit-cnn` is the one in use; verified working with
  PyTorch 2.5.1+cu121 against the Quadro RTX 8000.

## GPU utilization

First real-data baseline runs showed GPU utilization hovering around ~10%
and low power draw -- for a ~7-14k-parameter 1D-CNN this is expected unless
two things are tuned, both now wired up (previously `num_workers` was
declared per-profile in `configs/profile/*.yaml` but never actually passed to
any `DataLoader`, so it silently did nothing):

1. **`DataLoader(num_workers=...)`**: with `num_workers=0`, each batch is
   assembled synchronously on the main process, so the GPU sits idle while
   the CPU slices windows out of the EEG signal and builds the tensor.
   `configs/profile/server.yaml` sets `num_workers: 4`; `scripts/train.py`
   now threads `cfg.profile.num_workers` through `training/engine_baseline.py`
   into every `DataLoader`, with `pin_memory=True` + `non_blocking=True`
   transfers and `persistent_workers=True` so workers aren't respawned every
   epoch.
2. **Batch size**: `configs/train/default.yaml` raised `batch_size` from 64 to
   256. These models are small enough that even 256 samples take negligible
   VRAM (all four servers have 24-48GB); a bigger batch means fewer
   Python-level iterations and kernel launches per epoch, which is what
   actually drives GPU utilization up for a model this size -- not more VRAM
   usage.

Even with both fixes, expect GPU utilization to stay moderate (not
consistently >80%) for models this small -- that's inherent to the
architecture, not a misconfiguration. If throughput still matters after
this, the next lever is running multiple folds concurrently (e.g. two
`train.py` processes, one per model, as already done) or increasing batch
size further before reaching for anything more invasive.

Resume behavior: `scripts/train.py` now skips any fold whose
`<fold_id>.metrics.json` already exists (pass `train.force_retrain=true` to
redo it), so restarting a run after a config change never re-trains
already-completed folds.

## "Too many open files" -- two independent causes, both fixed

This has recurred twice with different root causes; both are now fixed in
code, but the underlying risk (many `DataLoader(num_workers>0)` constructions
over a 66-fold sweep) means it's worth understanding rather than just
re-raising `ulimit -n` each time:

1. **Tensor-sharing strategy** (first occurrence): torch's default
   `file_descriptor` strategy for passing tensors between worker processes
   consumes an fd per shared tensor; `DataLoader` iterators hold reference
   cycles that only get cleaned up on a full GC pass, so cleanup lagged
   creation across many folds. Fixed by
   `torch.multiprocessing.set_sharing_strategy("file_system")` in
   `scripts/train.py`.
2. **Worker/semaphore count from one-shot scoring loaders** (second
   occurrence, survived fix #1): every fold built up to 4 `DataLoader`s
   (train, val, and two one-shot scoring loaders for val/test), each with
   `num_workers=4` spawning 4 worker processes + OS-level locks/semaphores
   (and, under `file_system` sharing, a temp directory). The one-shot scoring
   loaders gained nothing from multiprocessing (a single pass, never
   iterated across epochs) but still paid this cost every fold. Fixed in
   `training/engine_baseline._score_partition`, which now always forces
   `num_workers=0` regardless of what's configured -- only the train/val
   loaders (genuinely iterated many times per fold) use worker processes.

Belt-and-suspenders: also raise the shell's open-file limit before a long
run, ideally in `~/.bashrc` so it doesn't need retyping every session:
`ulimit -n 65536`. If this recurs a third time with a different signature,
suspect the train/val loaders themselves (2 per fold, still using
`num_workers>0` + `persistent_workers=True`) rather than the scoring path.

## Other servers (for later scale-out)

- **SERVER-01**: same 48GB GPU class as SERVER-02, newer kernel (6.8) and
  driver (595.71.05); candidate if SERVER-02 becomes contended.
- **SERVER-03 / SERVER-04**: matched RTX 3090 (24GB) pair on Ubuntu 22.04 --
  the inventory doc's own suggestion is to use this pair for multi-seed
  ablation once the pipeline is validated on SERVER-02 (memo 5.3: three
  random seeds per training run maps naturally onto two matched boxes).
