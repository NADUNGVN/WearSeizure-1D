# Re-measuring the KV260 on a complete recording

For the hardware team. This is how to repeat, without help, the measurement
behind the Section V-A claim: the board computes exactly what the model
computes, over one full hour of EEG, and raises the same alarms.

## What the measurement shows, and what it does not

| Question | How it is answered | Result on 2026-11-02 |
|---|---|---|
| Does the board compute what the model computes? | Every byte of the final PL feature map (64 x 32 INT8) of every window, compared with the integer golden model | 3,597 / 3,597 windows identical |
| Used as the device, does it alarm the same? | The board's own feature maps taken through host GAP/FC, EMA and the frozen hysteresis | 3 / 3 seizures, delays 24, 25, 30 s, 0 false alarms, identical to software |
| How good is the detector? | **Not** measured on the board. Section IV, 185 h of test data | -- |

It is a reproduction check on one recording (`chb15_40`, 1.00 h, 3 seizures),
chosen because the board holds that fold's weights and the model detects its
seizures. It is not a sensitivity measurement. Per-window accuracy on a short
excerpt is not used: it depends on which excerpt is cut, and the device does
not decide per window.

## Before you start

* KV260-02 is **shared**. Another team's bitstream can be loaded even though
  the FPGA reports `operating`. The script checks this and stops.
* Stop `board_viewer.py` / `continuous_viewer.py`; only one program may drive
  the core.
* Our repository is at `/home/ubuntu/Manh/WearSeizure-1D` on the board and the
  servers. Use it read-only; pull nothing yourself unless asked.

## Step 1 -- on the board

```
ssh ubuntu@192.168.50.21
hostname; uname -m                      # KV260-02, aarch64
git -C /home/ubuntu/Manh/WearSeizure-1D log -1 --oneline
tmux new -s remeasure                   # survives a dropped SSH session
cd /home/ubuntu/Manh/WearSeizure-1D
sudo bash scripts/board_remeasure.sh; echo "exit=$?"
```

The script refuses, with a message, when: the machine is not the board; it is
not run with `sudo`; a viewer is running; the last bitstream written to the
FPGA is not `SoC_AI_wrapper.bit`; or your own driver fails your own pinned
window 000. Only then does it run the 3,597 windows (several minutes,
progress every 200 windows) into a new file:

```
/home/ubuntu/Anhquan/board_runs/board_run_chb15_40_<date>_<time>.json
```

Pass: `complete: 3597 windows`, a `sha256sum` line, `exit=0`.

If it stops on the bitstream, ask whoever loaded the other design, then:

```
cd /home/ubuntu/Anhquan/Embedded_Software
sudo sh load_bitstream.sh /home/ubuntu/Anhquan/SoC_AI_wrapper.bit; echo $?
sudo dmesg | tail -1                     # writing SoC_AI_wrapper.bit ...
```

and run Step 1 again.

## Step 2 -- on SERVER-02 (needs NumPy)

The board-run file is visible from the server over NFS. Write the table next
to it, not into our repository's `docs/`:

```
conda activate chbmit-cnn
cd /home/ubuntu/Manh/WearSeizure-1D
RUN=/home/ubuntu/Anhquan/board_runs/board_run_chb15_40_<date>_<time>.json
python scripts/hardware_verification_table.py --board-run $RUN \
  --markdown ${RUN%.json}_table.md --json ${RUN%.json}_table.json; echo "exit=$?"
```

It replays the golden model on all three complete recordings (a few minutes),
then compares every byte of your run. Pass: the `chb15_40` row reads

```
3,597/3,597 bit-exact; 3/3 detected (delay 24, 25, 30 s), 0 false alarms
```

followed by `PASS` and `exit=0`. A run that is incomplete, has any differing
byte, or alarms differently prints `FAIL` and exits 1; the `_table.json` names
the first differing windows (`board_feature_mismatches`).

## What may go into the paper

Only what the `PASS` table shows: the recording, the window count, "identical
in every byte", and the alarms (seizures detected, delays, false alarms).
Keep detection performance in Section IV. A paragraph matching the
2026-11-02 result:

```latex
The verification uses complete CHB-MIT recordings rather than excerpts, so that
the board is exercised with the same continuous stream and alarm logic as the
deployed device. For recording \texttt{chb15\_40} (1.00~h, 3,597 windows of 1024
samples with a 256-sample hop), the final PL feature map ($64\times32$, INT8)
returned by the board matches the integer golden model in every byte of every
window. Passing these feature maps through the host GAP/FC and the frozen
post-processor of Section~\ref{sec:method} raises the same alarms as the software
reference: all three seizures are detected, with delays of 24, 25 and 30~s, and
no false alarm occurs. The board therefore reproduces the model exactly, and the
detection performance reported in Section~\ref{sec:software-results} applies to
the hardware unchanged.
```

## When something goes wrong

| Message | Meaning | Do |
|---|---|---|
| `this is SERVER-02 (x86_64)` | Not on the board | `ssh ubuntu@192.168.50.21` |
| `last loaded with 'gemm_system_wrapper.bit'` (or any other) | Another team's design is on the FPGA | Ask, then reload `SoC_AI_wrapper.bit` |
| `Core did not enter LOAD` | Same cause, seen by the driver directly | As above |
| `board fails its own pinned check` | Your own driver/assets do not pass window 000 | Fix on your side first; nothing of ours ran |
| `board assets are fold ... these vectors are fold ...` | `Embedded_Software` now holds another fold's weights | The check only exists for `chb15__chb15_40` today |
| `... changed since the board ran it` (Step 2) | Our vectors changed after your run | Re-run Step 1 |

## Not covered yet

`chb01_03` (1 h, 1 seizure) and `chb04_28` (4 h, 2 seizures) need their own
folds' weights in your packed format (`packed_weights.hex`, `packed_biases.hex`,
`instructions.hex`). Until then the board column of those rows reads
`not yet measured`.
