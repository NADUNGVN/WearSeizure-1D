#!/usr/bin/env bash
# Re-measure the KV260 on the complete chb15_40 recording, from one command.
#
# Runs ON THE BOARD, as root (the driver maps /dev/mem):
#
#   cd /home/ubuntu/Manh/WearSeizure-1D
#   sudo bash scripts/board_remeasure.sh
#
# What it does, and refuses
# -------------------------
# 1. Refuses unless this is the board (aarch64).
# 2. Refuses unless the bitstream most recently written to the FPGA is the
#    accelerator's. KV260-02 is shared: another team's bitstream leaves the
#    FPGA "operating", and the driver then fails with nothing more than
#    "Core did not enter LOAD". Loading the bitstream is left to the person
#    running this, because it evicts whatever design is on the board.
# 3. Runs the RTL team's own driver on their own pinned window 000 and
#    requires it to match their golden output: if the board fails its own
#    check, our measurement would only measure that failure.
# 4. Runs every window of chb15_40 (3,597) through the board with
#    scripts/board_run_recording.py, into a new time-stamped file, never
#    overwriting an earlier one.
#
# The board run is scored afterwards on a machine with NumPy -- see
# docs/BOARD_REMEASURE.md.
#
# Environment (defaults in brackets):
#   EMBEDDED_DIR  the RTL team's build  [/home/ubuntu/Anhquan/Embedded_Software]
#   OUT_DIR       where runs are kept   [/home/ubuntu/Anhquan/board_runs]
#   BITSTREAM     expected .bit name    [SoC_AI_wrapper.bit]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EMBEDDED_DIR="${EMBEDDED_DIR:-/home/ubuntu/Anhquan/Embedded_Software}"
OUT_DIR="${OUT_DIR:-/home/ubuntu/Anhquan/board_runs}"
BITSTREAM="${BITSTREAM:-SoC_AI_wrapper.bit}"
VECTORS="hardware/export_chb15__chb15_40/whole_recording/chb15_40_sz0"
PINNED="$EMBEDDED_DIR/onset_vectors/chb15_40_sz0/windows/000"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ABORT: this is $(hostname) ($(uname -m)); run it on the KV260 (ssh ubuntu@192.168.50.21)."
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "ABORT: the driver needs /dev/mem; run it as: sudo bash scripts/board_remeasure.sh"
  exit 1
fi
if pgrep -f "board_viewer|continuous_viewer|verify_board_model" >/dev/null; then
  echo "ABORT: another program is driving the core; stop it first:"
  pgrep -af "board_viewer|continuous_viewer|verify_board_model"
  exit 1
fi

loaded="$(dmesg | grep "fpga_manager fpga0: writing" | tail -1 | sed 's/.*writing \(.*\) to .*/\1/' || true)"
if [ "$loaded" != "$BITSTREAM" ]; then
  echo "ABORT: the FPGA was last loaded with '${loaded:-unknown}', not '$BITSTREAM'."
  echo "Ask whoever loaded it before replacing it, then:"
  echo "  cd $EMBEDDED_DIR && sudo sh load_bitstream.sh /home/ubuntu/Anhquan/$BITSTREAM"
  exit 2
fi
echo "bitstream: $loaded"

echo "pre-check: the RTL team's driver on their pinned window 000"
pre="$("$EMBEDDED_DIR/board_smoke_core" --run --json --assets-dir "$EMBEDDED_DIR" \
        --input "$PINNED/00_input.txt" --expected "$PINNED/13_context_1_pw.txt" \
        --input-p 3 --timeout-ms 2000 2>&1 || true)"
if ! printf '%s\n' "$pre" | grep -q '"output_match":true'; then
  echo "ABORT: the board fails its own pinned check; nothing of ours was run."
  printf '%s\n' "$pre" | grep -v '^{' | tail -5
  exit 3
fi
echo "pre-check: PASS"

mkdir -p "$OUT_DIR"
out="$OUT_DIR/board_run_chb15_40_$(date +%Y%m%d_%H%M%S).json"
cd "$REPO"
echo "revision: $(git -C "$REPO" log -1 --oneline 2>/dev/null || echo unknown)"
status=0
/usr/bin/python3 -I scripts/board_run_recording.py --embedded-dir "$EMBEDDED_DIR" \
  --vectors "$VECTORS" --output "$out" || status=$?
if [ -n "${SUDO_UID:-}" ] && [ -f "$out" ]; then
  chown "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$OUT_DIR" "$out"
fi
[ -f "$out" ] && sha256sum "$out"
if [ "$status" -ne 0 ]; then
  echo "FAILED (exit $status); the partial run is kept in $out"
  exit "$status"
fi
echo
echo "Done. Score it on SERVER-02 (see docs/BOARD_REMEASURE.md):"
echo "  python scripts/hardware_verification_table.py --board-run $out \\"
echo "    --markdown ${out%.json}_table.md --json ${out%.json}_table.json"
