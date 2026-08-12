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
- Known clutter to clean up before use: three old, unsuccessful experiment
  directories under `~/Manh` --
  `1D-CNN-Accelerator-for-EEG_Detection`,
  `1D-CNN-Accelerator-for-EEG_Detection-q1`,
  `1D-CNN-Accelerator-for-EEG_Detection-main`.
  See `scripts/server_bootstrap.sh` for the review-then-delete commands
  (deletion is left as an explicit, manually-run step, never automated).

## Other servers (for later scale-out)

- **SERVER-01**: same 48GB GPU class as SERVER-02, newer kernel (6.8) and
  driver (595.71.05); candidate if SERVER-02 becomes contended.
- **SERVER-03 / SERVER-04**: matched RTX 3090 (24GB) pair on Ubuntu 22.04 --
  the inventory doc's own suggestion is to use this pair for multi-seed
  ablation once the pipeline is validated on SERVER-02 (memo 5.3: three
  random seeds per training run maps naturally onto two matched boxes).
