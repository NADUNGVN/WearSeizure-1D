"""Continuous test vectors spanning a seizure onset, for RTL bring-up.

    python scripts/export_onset_vectors.py profile=server data=chbmit \
        +export.pre_s=30 +export.post_s=60

The sixteen vectors already in `hardware/export/test_vectors/` are a per-layer
dump of ONE window. They answer "does the datapath compute the right numbers",
and nothing else. They cannot answer the questions that break a bring-up:

* does the accelerator stream continuously for a minute without drifting,
* does the line buffer hold its state across window boundaries,
* does the postprocessor's EMA, hysteresis and run-length raise an alarm where
  the software does,
* and does any of it behave differently on the transition from interictal to
  ictal, which is the only part of the recording the device exists for.

This writes a continuous excerpt around a real seizure onset: the int8 sample
stream the accelerator consumes, and the expected output for every window that
stream contains, from the same DFP8 golden model the RTL must match bit for
bit.

**The model is off-chip up to the AXI-Stream port.** The causal 1-30 Hz
band-pass and the affine normaliser run on the PS, so what this emits is
already filtered and normalised. The RTL is not expected to reproduce them.

**The weights decide the recording.** `hardware/export/manifest.json` records
the fold its weights came from, and that fold's test partition is the only
recording this model has not seen. Streaming a different one would compare the
hardware against logits from a model that was trained on the data. The script
refuses to do that.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.models.factory import build_model
from wearseizure.postprocess.ema import ema_smooth
from wearseizure.postprocess.hysteresis import PostprocessParams
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing

log = get_logger(__name__)
bootstrap_env(sys.argv)

REPO = Path(__file__).resolve().parent.parent


def write_lf(path: Path, text: str) -> None:
    """LF regardless of platform: `$readmemh` on a CRLF file reads garbage."""
    path.write_bytes(text.encode("utf-8"))


def load_golden(model_dir: Path):
    """Import the delivered golden model by path, not by package.

    `hardware/golden_model.py` is deliberately standalone -- it is the file the
    RTL team received -- so it is not importable as part of this package.
    Registering it in `sys.modules` before executing it is what lets its own
    dataclasses resolve.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "golden_model", REPO / "hardware" / "golden_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["golden_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def margin_for_probability(p: float, p_out: int) -> int:
    """The smallest integer logit margin whose probability reaches `p`.

    The thresholds were frozen on validation as probabilities, but the hardware
    never computes a softmax -- it has two int8 logits and their difference.
    For two classes the softmax is a logistic in that difference, so
    `prob >= p` is exactly `margin_int >= 2**p_out * log(p / (1 - p))`, and the
    comparison the RTL needs is an integer one. Rounded UP so the integer test
    is never looser than the float test it stands in for.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"threshold {p} is not a probability")
    return int(math.ceil((2.0 ** p_out) * math.log(p / (1.0 - p))))


def probability_from_margin(margin, p_out: int) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(margin, dtype=np.float64) / (2.0 ** p_out)))


def readme_for(c: dict, bits: int) -> str:
    """Prose for whoever opens the directory without this script to hand."""
    e, w, pp, s = c["excerpt"], c["windows"], c["postprocess"], c["scoring"]
    step_pp = 100.0 * s["probability_step_at_margin_0"]
    hyst = ("" if pp.get("hysteresis_survives_integer_thresholds", True) else
            "\n> **The two thresholds map to the same integer margin.** On this datapath\n"
            "> the hysteresis is not a hysteresis -- it is a single threshold. Fix that\n"
            "> before using these vectors to validate the postprocessor.\n")
    return f"""# Onset test vectors -- {c['recording']}, seizure {c['seizure_index']}

{e['duration_s']:.1f} s of continuous signal straddling a real seizure onset:
**{e['seconds_before_onset']:.0f} s before** it and **{e['seconds_after_offset']:.0f} s
after the seizure ends**. {w['count']} windows, {w['ictal_count']} of them ictal.
The onset is at **{c['onset_sec']:.1f} s** into the recording and the offset at
**{c['offset_sec']:.1f} s**.

The point of the excerpt is the transition. A design can score an interictal
hour correctly and still fail at the only moment it exists for, and a per-layer
dump of one window cannot show that.

## Files

| file | what it is |
|---|---|
| `stream_input.hex` | {e['samples']} samples, int{bits} two's complement, one hex byte per line. This is the AXI-Stream payload. |
| `stream_input.csv` | the same samples with time, time relative to onset, and the float value they were quantised from. |
| `windows.csv` | one row per window: expected logits, margin, probability, and whether the postprocessor is above each threshold. |
| `contract.json` | every constant above in machine-readable form. |

## The arithmetic contract

* A sample's real value is `q / 2**{c['stream']['input_exponent']}`.
* The filter is **not** the accelerator's job: the causal 1-30 Hz band-pass and
  the affine normaliser run on the PS, and this stream is their output.
* Windows are {w['window_samples']} samples, advancing {w['stride_samples']}
  samples. Window *n* of `windows.csv` starts at `stream_offset_samples`.
* The hardware has **no softmax**. The decision rides on
  `margin = logit1 - logit0` as integers; the probability column is derived,
  shown only so the thresholds are legible.

## Where this will bite

**Score resolution.** One LSB of margin is **{step_pp:.3g} percentage points**
of probability at the decision boundary. The thresholds were frozen on a
continuous score, so in integer terms they are
`margin >= {pp['threshold_on_margin_int']}` (on) and
`margin >= {pp['threshold_off_margin_int']}` (off). Compare integers; do not
reconstruct the probability in hardware.
{hyst}
**EMA warm start.** `ema_probability` is the recursion run from the start of
the whole recording, not from this excerpt. Seed the testbench with
`ema_warm_start_probability = {pp['ema_warm_start_probability']:.8f}` or the
column will not match.

**EMA operand.** The filter smooths **probabilities**, not margins. The
logistic is non-linear, so an EMA on the margin is a different filter and will
move the alarms.

## Provenance

Weights from fold `{c['provenance']['fold_id']}`, seed {c['provenance']['seed']},
tag `{c['provenance']['run_tag']}` -- the same export the RTL is built against.
This recording is that fold's held-out test partition, so the model has never
seen it.
"""


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    ex = cfg.get("export", {})
    bits = int(ex.get("bits", 8))
    model_dir = Path(ex.get("model_dir", REPO / "hardware" / "export"))
    out_dir = Path(ex.get("vectors_out", model_dir / "onset_vectors"))
    pre_s = float(ex.get("pre_s", 30.0))
    post_s = float(ex.get("post_s", 60.0))
    per_window_files = bool(ex.get("per_window_files", False))
    which_seizure = ex.get("seizure", None)   # None = every seizure in the EDF

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}. Run export_dfp_hardware.py first; "
            "this script streams data THROUGH an exported model, it does not make one."
        )
    hw = json.loads(manifest_path.read_text(encoding="utf-8"))
    prov = hw["provenance"]
    p_out_fc = int(hw["layers"][-1]["p_out"])
    p_in = int(hw["input_exponent"])
    fs_manifest = int(hw["sampling_rate_hz"])
    window_samples = int(hw["window_samples"])

    # ---------------------------------------------------------------- data
    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    records = load_records_from_manifest(
        manifest_df,
        data_dir=cfg.data.generated_dir if cfg.data.name == "synthetic" else None,
        raw_dir=cfg.data.raw_dir if cfg.data.name != "synthetic" else None,
    )
    folds = load_folds(str(Path(cfg.split.folds_path)),
                       expected_manifest_hash=hash_manifest(manifest_df))
    by_id = {f.fold_id: f for f in folds}
    if prov["fold_id"] not in by_id:
        raise SystemExit(
            f"the exported weights come from fold {prov['fold_id']}, which is not in "
            f"{cfg.split.folds_path}. Either the splits were regenerated or the "
            "manifest is from another experiment; refusing to pair them."
        )
    fold = by_id[prov["fold_id"]]

    seed = seeds_from_cfg(cfg)[0]
    run_tag = run_tag_from_cfg(cfg)
    if (seed, run_tag) != (prov["seed"], prov["run_tag"]):
        raise SystemExit(
            f"the exported weights are seed {prov['seed']} / tag {prov['run_tag']}, but "
            f"this run is seed {seed} / tag {run_tag}. The float reference and the "
            "frozen thresholds would come from a different checkpoint than the "
            "integer weights. Pass the matching seed and tag."
        )

    art = Path(cfg.profile.artifacts_dir)
    src = fold_run_dir(art, cfg.model.name, cfg.split.name, cfg.window.name,
                       seed, tag=run_tag)
    ckpt = src / f"{fold.fold_id}.pt"
    metrics_path = src / f"{fold.fold_id}.metrics.json"
    for path in (ckpt, metrics_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; this fold has not been trained here.")

    window_s, stride_s = cfg.window.window_s, cfg.window.stride_s
    datasets, _band, _norm = build_fold_datasets(records, fold, window_s, stride_s)
    normalised = datasets["test"].records

    test_ids = sorted(fold.test_edf_ids)
    if len(test_ids) != 1:
        log.warning(f"{len(test_ids)} test EDFs in this fold; using the first with a seizure")
    edf_id = next((e for e in test_ids if records[e].meta.seizure_events), None)
    if edf_id is None:
        raise SystemExit(
            f"no seizure in the test partition of fold {fold.fold_id} "
            f"({', '.join(test_ids)}). There is no onset to straddle."
        )
    meta = records[edf_id].meta
    fs = int(round(meta.fs_hz))
    if fs != fs_manifest:
        raise SystemExit(f"{edf_id} is {fs} Hz, the manifest says {fs_manifest} Hz")
    signal = np.asarray(normalised[edf_id].signal, dtype=np.float64).ravel()
    stride_len = round(stride_s * fs)

    # ------------------------------------------------------------- model
    golden_mod = load_golden(model_dir)
    specs = [golden_mod.LayerSpec.from_manifest(d) for d in golden_mod.all_layers(hw)]
    weights = golden_mod.load_weights(model_dir / "weights", specs, bits=bits)
    golden = golden_mod.GoldenModel(hw, weights, bits=bits, check_overflow=True)

    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval().double()

    params_raw = json.loads(
        metrics_path.read_text(encoding="utf-8"))["frozen_postprocess"]["params"]
    params = PostprocessParams(**params_raw)

    # The decision resolution the hardware actually has. Two int8 logits at
    # exponent p_out mean the margin moves in steps of 1, and one step near the
    # boundary is worth a large slice of probability -- at p_out = 2 it is six
    # percentage points. The thresholds were frozen as probabilities on a
    # continuous score, so they can land inside a step the hardware cannot
    # express, and `threshold_off < threshold_on` can become
    # `margin_off == margin_on`. Hysteresis would then silently degrade to a
    # single threshold: no error, no warning, just a different detector.
    m_on = margin_for_probability(params.threshold_on, p_out_fc)
    m_off = margin_for_probability(params.threshold_off, p_out_fc)
    step_pp = 100.0 * (probability_from_margin([1], p_out_fc)[0] - 0.5)
    print(f"\nscore resolution: one LSB of margin is {step_pp:.3g} pp of probability "
          f"at the boundary")
    print(f"  threshold_on  {params.threshold_on:.4f} -> margin_int >= {m_on}")
    print(f"  threshold_off {params.threshold_off:.4f} -> margin_int >= {m_off}")
    if params.method == "hysteresis_runlength" and m_off >= m_on:
        print("  WARNING: both thresholds map to the same integer margin. On this "
              "datapath the\n  hysteresis collapses to a plain threshold. Either widen "
              "the gap, or carry more\n  score resolution into the postprocessor -- the "
              "48-bit accumulator before the\n  final requantisation has it.")

    # Quantise the WHOLE recording once. Every window below is a slice of this
    # array rather than a separately quantised copy, which is the point: it is
    # the guarantee that a testbench replaying the byte stream presents each
    # window exactly the bytes its expected output was computed from.
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    scaled = np.round(signal * (2.0 ** p_in))
    quantised = np.clip(scaled, lo, hi).astype(np.int64)
    n_clipped = int(np.count_nonzero((scaled < lo) | (scaled > hi)))

    # Score every window of the recording, not only the excerpt. The EMA is
    # causal and recursive, so its value at the excerpt's first window depends
    # on the whole hour before it. A testbench that starts the EMA at zero gets
    # different alarms than the software did, and the only way to hand over a
    # correct warm-start value is to have run the hour.
    n = len(quantised)
    starts = list(range(0, n - window_samples + 1, stride_len))
    log.info(f"{edf_id}: {n} samples, {len(starts)} windows -- scoring all of them "
             "so the EMA warm start is real")
    margins = np.empty(len(starts), dtype=np.int64)
    logits_i = np.empty((len(starts), 2), dtype=np.int64)
    for i, s in enumerate(starts):
        out, _ = golden.run(quantised[s:s + window_samples])
        logits_i[i] = out[:, 0]
        margins[i] = int(out[1, 0]) - int(out[0, 0])
        if i and i % 500 == 0:
            log.info(f"  {i}/{len(starts)}")
    probs = probability_from_margin(margins, p_out_fc)
    smoothed = ema_smooth(probs, params.ema_alpha) if params.ema_alpha else probs
    end_sec = np.array([(s + window_samples) / fs for s in starts])
    alarms = run_postprocess(end_sec, probs, params)

    # Which series the thresholds are compared against is method-dependent, and
    # getting it wrong is invisible: both series live in [0, 1] and move
    # together, so a testbench checking the raw score against threshold_on
    # produces plausible alarms in the wrong places. `run_postprocess` smooths
    # first for every method except `raw_threshold`, so this mirrors it rather
    # than guessing.
    decided_on = probs if params.method == "raw_threshold" else smoothed

    # ------------------------------------------------------------ excerpts
    events = sorted(meta.seizure_events, key=lambda e: e.onset_sec)
    chosen = events if which_seizure is None else [events[int(which_seizure)]]
    ensure_dir(out_dir)
    index: list[dict] = []
    digits, mask = bits // 4, (1 << bits) - 1

    for k, event in enumerate(chosen):
        onset_idx = int(round(event.onset_sec * fs))
        offset_idx = int(round(event.offset_sec * fs))
        # Snap the excerpt to the window grid, so window boundaries inside the
        # stream fall where the evaluation put them. An excerpt that starts
        # off-grid produces windows no software run ever scored.
        want_start = onset_idx - int(round(pre_s * fs))
        start = max(0, (want_start // stride_len) * stride_len)
        stop = min(n, offset_idx + int(round(post_s * fs)))
        if stop - start < window_samples:
            log.warning(f"seizure {k} excerpt is shorter than one window; skipping")
            continue

        seg = quantised[start:stop]
        seg_f = signal[start:stop]
        tag = f"{edf_id}_sz{k}"
        sub = ensure_dir(out_dir / tag)

        write_lf(sub / "stream_input.hex",
                 "".join(f"{int(v) & mask:0{digits}X}\n" for v in seg))

        rows = ["sample_idx,t_sec,t_rel_onset_s,x_normalised,x_int8"]
        for j, (q, x) in enumerate(zip(seg, seg_f)):
            t = (start + j) / fs
            rows.append(f"{start + j},{t:.6f},{t - event.onset_sec:+.6f},"
                        f"{x:.8f},{int(q)}")
        write_lf(sub / "stream_input.csv", "\n".join(rows) + "\n")

        # Windows fully inside the excerpt, kept in the recording's own
        # indexing so a row here can be found again in the full-run results.
        sel = [i for i, s in enumerate(starts)
               if s >= start and s + window_samples <= stop]
        if not sel:
            log.warning(f"seizure {k}: no complete window inside the excerpt")
            continue

        warm = float(smoothed[sel[0] - 1]) if sel[0] > 0 else 0.0
        hdr = ("window_idx,stream_offset_samples,start_sec,end_sec,"
               "end_rel_onset_s,label,logit0_int,logit1_int,margin_int,"
               "probability,ema_probability,decision_score,above_on,above_off")
        wrows = [hdr]
        n_ictal = 0
        for i in sel:
            s = starts[i]
            es = end_sec[i]
            label = int(any(e.onset_sec < es and (s / fs) < e.offset_sec for e in events))
            n_ictal += label
            wrows.append(
                f"{i},{s - start},{s / fs:.4f},{es:.4f},{es - event.onset_sec:+.4f},"
                f"{label},{logits_i[i][0]},{logits_i[i][1]},{margins[i]},"
                f"{probs[i]:.8f},{smoothed[i]:.8f},{decided_on[i]:.8f},"
                f"{int(decided_on[i] >= params.threshold_on)},"
                f"{int(decided_on[i] >= params.threshold_off)}")
        write_lf(sub / "windows.csv", "\n".join(wrows) + "\n")

        if per_window_files:
            wdir = ensure_dir(sub / "windows")
            for i in sel:
                s = starts[i]
                write_lf(wdir / f"win_{i - sel[0]:04d}.hex",
                         "".join(f"{int(v) & mask:0{digits}X}\n"
                                 for v in quantised[s:s + window_samples]))

        contract = {
            "recording": edf_id,
            "seizure_index": k,
            "onset_sec": event.onset_sec,
            "offset_sec": event.offset_sec,
            "excerpt": {
                "start_sample": start, "stop_sample": stop,
                "duration_s": (stop - start) / fs,
                "samples": int(stop - start),
                "seconds_before_onset": (onset_idx - start) / fs,
                "seconds_after_offset": (stop - offset_idx) / fs,
            },
            "stream": {
                "format": f"int{bits}, two's complement, one hex byte per line",
                "input_exponent": p_in,
                "real_value": "x = q / 2**input_exponent",
                "sampling_rate_hz": fs,
                "pre_filtered": ("causal 1-30 Hz band-pass and affine normalisation "
                                 "run on the PS; this stream is their output"),
                "clipped_samples_whole_recording": n_clipped,
            },
            "windows": {
                "window_samples": window_samples,
                "stride_samples": stride_len,
                "count": len(sel),
                "first_window_idx": sel[0],
                "ictal_count": n_ictal,
            },
            "scoring": {
                "logit_exponent": p_out_fc,
                "margin": "logit1 - logit0, as integers; no softmax in hardware",
                "probability": "1 / (1 + exp(-margin / 2**logit_exponent))",
                "probability_step_at_margin_0": float(
                    probability_from_margin([1], p_out_fc)[0] - 0.5),
            },
            "postprocess": {
                **params_raw,
                "threshold_on_margin_int": m_on,
                "threshold_off_margin_int": m_off,
                "hysteresis_survives_integer_thresholds": m_off < m_on,
                "ema_warm_start_probability": warm,
                "decision_score_column": (
                    "probability" if params.method == "raw_threshold"
                    else "ema_probability"),
                "ema_note": ("the EMA smooths PROBABILITIES, not margins. Smoothing the "
                             "margin instead is not the same filter -- the logistic is "
                             "non-linear -- and will not reproduce these alarms."),
                "warm_start_note": ("the recursion ran from the start of the recording. "
                                    "A testbench that begins the excerpt with the EMA at "
                                    "zero will not match the ema_probability column."),
            },
            "alarms_whole_recording": [
                {"start_sec": a, "end_sec": b} for a, b in alarms],
            "provenance": prov,
        }
        write_lf(sub / "contract.json", json.dumps(contract, indent=2) + "\n")
        write_lf(sub / "README.md", readme_for(contract, bits))
        index.append({"dir": tag, "onset_sec": contract["onset_sec"],
                      "offset_sec": contract["offset_sec"],
                      "excerpt": contract["excerpt"],
                      "windows": contract["windows"]})

        # What the excerpt is actually worth, said out loud.
        fires = [i for i in sel if decided_on[i] >= params.threshold_on]
        print(f"\n{tag}: {(stop - start) / fs:.1f} s, {len(sel)} windows "
              f"({n_ictal} ictal)")
        print(f"  onset at {event.onset_sec:.1f} s; excerpt holds "
              f"{(onset_idx - start) / fs:.0f} s before it and "
              f"{(stop - offset_idx) / fs:.0f} s after the seizure ends")
        if not fires:
            print("  the DFP8 model never crosses threshold_on inside this excerpt.")
            print("  That is a real miss on this fold, not a broken export. It still")
            print("  works as a negative test; pick another seizure for a firing one.")
        else:
            print(f"  first window over threshold_on: "
                  f"{end_sec[fires[0]] - event.onset_sec:+.1f} s relative to onset")

    write_lf(out_dir / "index.json", json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {len(index)} excerpt(s) under {out_dir}")
    print(f"peak accumulator over the whole recording: {golden.peak_acc} "
          f"({golden.peak_acc.bit_length()} bits of {golden_mod.ACC_BITS})")


if __name__ == "__main__":
    main()
