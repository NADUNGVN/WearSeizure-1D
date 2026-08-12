# Architecture

Software pipeline for WearSeizure-1D (memo section 4). Everything below runs
today against synthetic data on a CPU; the same code runs against real
CHB-MIT data on the training server by switching `profile=server`.

## Dataflow

```
EDF/synthetic -> manifest -> split (per-EDF/subject) -> causal filter + affine
normalize (fit on train) -> windowing -> model -> postprocess -> event metrics
```

Each arrow is a module boundary in `src/wearseizure/`:

| Stage | Module | Notes |
|---|---|---|
| Manifest | `data/manifest.py` | Schema + content hash; Appendix A channel map |
| Real EDF loading | `data/io_edf.py`, `data/chbmit_summary_parser.py` | Server profile only |
| Synthetic data | `data/synthetic.py` | Local profile; same manifest schema |
| Split | `data/splits.py` | Patient-specific LOSO-EDF + zero-shot LOSO-subject |
| Filter/normalize | `signal/filters.py`, `signal/normalize.py` | Causal, fit on train only |
| Windowing | `data/windowing.py` | Causal, fold-scoped, raises on leakage |
| Dataset | `data/dataset.py`, `data/sampler.py` | Per-fold, per-partition |
| Models | `models/layers.py`, `models/wearseizure1d.py`, `models/baselines.py` | Table 4 + reproduction baselines |
| Training | `training/loop.py`, `training/engine_baseline.py` | Shared by all models |
| Threshold freeze | `training/threshold_selection.py` | Val-only, frozen before test |
| Postprocess | `postprocess/ema.py`, `postprocess/hysteresis.py`, `postprocess/pipeline.py` | EMA + hysteresis + run-length + merge |
| Evaluation | `eval/event_matching.py`, `eval/metrics_event.py`, `eval/metrics_segment.py`, `eval/bootstrap.py`, `eval/report.py` | Event-primary, segment-secondary |
| Quantization | `quant/scales.py`, `quant/qat.py`, `quant/ptq.py`, `quant/int_reference.py` | INT8 W/A, INT32 accumulator |
| RTL interface (spec only) | `rtl_interface/golden_io_contract.py`, `rtl_interface/spec.md` | No RTL yet; interface contract only |

## WearSeizure-1D (Table 4)

Stem (Conv1D k7,s2) -> B1 (DW k5,s2) -> B2/B3/B4 (multi-scale
`[DW k3 || DW k5,dilation]`, s2, dilation 1/2/4) -> Context (2x dilated
depthwise-separable, s1) -> GAP -> FC(2 logits). Hard budget: <=32k params,
<=2M MACs (Gate G1); target 13,810 params / 0.644M MACs. See
`models/wearseizure1d.py` and `tests/unit/test_models_shapes_and_budget.py`.

## Why synthetic data exists

The local machine has no CHB-MIT data and no GPU. `data/synthetic.py`
generates EEG-shaped signals (causally filtered noise + inserted ictal
patterns + artifacts) with the exact same manifest schema as real data, so
every module above can be exercised end-to-end on a laptop CPU in seconds.
Synthetic results carry no clinical meaning -- `profile.enforce_gates=false`
under `profile=local_synthetic` reflects that explicitly.
