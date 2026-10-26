# Audit recorded FPGA alarm evidence

`scripts/audit_fpga_alarms.py` is a standalone, standard-library-only reader of
the deployed application's schema-v2 `board_verification.json`. It never imports
that application, runs its driver, programs the FPGA, or searches thresholds.
Deliver this script through the **WearSeizure-1D Git repository**. Run it from
that checkout, outside `Embedded_Software`; do not copy files into the pinned
application, change its assets, or update the separate student RTL repository.

The audit checks the report against current asset hashes, pinned source files,
every recorded INT8 feature map, independently recomputed host GAP/FC, and the
exported logits. It recomputes EMA on probabilities with the contract's warm
start and uses frozen causal hysteresis/run-length/merge parameters.

## Update and run on KV260-02

Use the existing board shell. Do not start another `bash`, activate conda,
use `sudo`, or run the hardware driver for this audit. First check that the
WearSeizure-1D checkout exists and has no unexpected local changes:

```
git -C /home/ubuntu/Manh/WearSeizure-1D status --short --branch
```

Preserve local changes. Once the checkout is confirmed safe to update, pull
only our repository:

```
git -C /home/ubuntu/Manh/WearSeizure-1D pull --ff-only origin main
```

Check the exit status immediately with `echo $?`, then check the revision:

```
git -C /home/ubuntu/Manh/WearSeizure-1D log -1 --oneline
```

It must match the reviewed revision delivered with these instructions. If the
path is absent or the revision differs, stop and check before running. The
server and board may share this checkout, but that must not be assumed.

Run the read-only audit using the board's system Python:

```
/usr/bin/python3 -I /home/ubuntu/Manh/WearSeizure-1D/scripts/audit_fpga_alarms.py --embedded-dir /home/ubuntu/Anhquan/Embedded_Software --output /home/ubuntu/Anhquan/alarm_audit_01.json
```

Check the exit status immediately with `echo $?` (zero means success). Inspect
the new JSON without importing anything from the deployed application:

```
/usr/bin/python3 -I -m json.tool /home/ubuntu/Anhquan/alarm_audit_01.json
```

Check the new JSON's `passed`, `scope`, `windows`,
`feature_bytes_matched`, `anchor_event`, and `event_metrics`. Existing outputs
are refused rather than overwritten. A failure is not a reason to change test
thresholds: inspect the named inconsistency first.

An excerpt uses reference prehistory for EMA and the initial off-state. Its
anchor detection and clipped alarm are diagnostics, not a cohort sensitivity
or an estimate of clinical FAR. The right-censored alarm flag matters when the
excerpt stops while an alarm is active.

For complete-recording metrics, add `--summary /path/to/chbXX-summary.txt`.
The original summary must match the recording, all seizure annotations and CSV
labels, and the recording's full duration. Otherwise no complete-recording
event sensitivity/FAR is reported. The current short board demo report is not
turned into a whole-recording evaluation by supplying a summary.

This audits saved evidence, not the currently loaded bitstream. A locally
hashed `.bit` file does not establish the identity of the image in PL.

## Local Windows validation

From the Windows repository root, tests include production-hysteresis parity
and negative evidence cases:

```
.\.venv\Scripts\python.exe -B -m pytest tests/unit/test_audit_fpga_alarms.py -q -p no:cacheprovider
```

In PowerShell, check `$LASTEXITCODE` immediately for zero and confirm no
failures. Test fixtures are synthetic and
must never be presented as measured FPGA or clinical results.
