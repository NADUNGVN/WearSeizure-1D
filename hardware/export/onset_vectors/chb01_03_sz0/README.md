# Onset test vectors -- chb01_03, seizure 0

130.0 s of continuous signal straddling a real seizure onset:
**30 s before** it and **60 s
after the seizure ends**. 127 windows, 43 of them ictal.
The onset is at **2996.0 s** into the recording and the offset at
**3036.0 s**.

The point of the excerpt is the transition. A design can score an interictal
hour correctly and still fail at the only moment it exists for, and a per-layer
dump of one window cannot show that.

**The model detects this seizure**: the alarm raises +9.0 s after onset, of which 2.0 s is the `run_length = 3` requirement rather than the detector hesitating. 0 false alarm(s) inside the excerpt.

## Files

| file | what it is |
|---|---|
| `stream_input.hex` | 33280 samples, int8 two's complement, one hex byte per line. This is the AXI-Stream payload. |
| `stream_input.csv` | the same samples with time, time relative to onset, and the float value they were quantised from. |
| `windows.csv` | one row per window: expected logits, margin, probability, and whether the postprocessor is above each threshold. |
| `contract.json` | every constant above in machine-readable form. |

## The arithmetic contract

* A sample's real value is `q / 2**3`.
* The filter is **not** the accelerator's job: the causal 1-30 Hz band-pass and
  the affine normaliser run on the PS, and this stream is their output.
* Windows are 1024 samples, advancing 256
  samples. Window *n* of `windows.csv` starts at `stream_offset_samples`.
* The hardware has **no softmax**. The decision rides on
  `margin = logit1 - logit0` as integers; the probability column is derived,
  shown only so the thresholds are legible.

## Where this will bite

**Score resolution.** One LSB of margin is **6.22 percentage points**
of probability at the decision boundary. The thresholds were frozen on a
continuous score, so in integer terms they are
`margin >= -3` (on) and
`margin >= -18` (off). Compare integers; do not
reconstruct the probability in hardware.

**EMA warm start.** `ema_probability` is the recursion run from the start of
the whole recording, not from this excerpt. Seed the testbench with
`ema_warm_start_probability = 0.00326489` or the
column will not match.

**EMA operand.** The filter smooths **probabilities**, not margins. The
logistic is non-linear, so an EMA on the margin is a different filter and will
move the alarms.

## Provenance

Weights from fold `chb01__chb01_03`, seed 0,
tag `L8` -- the same export the RTL is built against.
This recording is that fold's held-out test partition, so the model has never
seen it.
