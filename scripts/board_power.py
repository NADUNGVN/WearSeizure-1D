"""Measure what one inference costs on the KV260, differentially.

Runs ON THE BOARD, as root, with the system Python and the standard library
only -- the board has no NumPy, and the accelerator run it measures does not
need any either.

    sudo /usr/bin/python3 -I scripts/board_power.py \\
        --out /home/ubuntu/Anhquan/power_chb15_40.json \\
        --inferences 3597 \\
        -- /usr/bin/python3 -I scripts/board_run_recording.py \\
             --embedded-dir /home/ubuntu/Anhquan/Embedded_Software \\
             --vectors hardware/export_chb15__chb15_40/whole_recording/chb15_40_sz0 \\
             --output /home/ubuntu/Anhquan/board_run_power.json

Why differential, and what the number is
----------------------------------------
The KV260 exposes **one** INA260 at I2C 0x40, on **VCC_SOM** -- the whole
module, PS and PL on the same rail. There is no per-rail monitor on K26, so
absolute PL power is not obtainable on this board at any effort. What is
obtainable is the difference between the board doing nothing and the board
running inferences: PS static, PL static, DDR and the fan are present in both
and cancel.

So the result is **dynamic energy per inference for the whole SOM**, and it
includes the CPU's share of the per-window work -- the DMA in, the polling, the
2,048-byte readback, and GAP and FC which run on the PS. That is the honest
system figure. It is **not** PL-only accelerator energy, and this script will
not let a report call it that.

The noise floor comes first
---------------------------
Baseline is measured **twice**, before and after the workload, and their spread
is reported. A difference smaller than that spread is not a measurement, and
the script says so instead of printing a number. Thermal drift over a long run
shows up as a gap between the two baselines; that gap is reported too, because
a rising idle floor otherwise reads as rising dynamic power.

What is NOT measured here
-------------------------
Clock-gated idle. The accelerator has no clock gate exposed, so "idle" is the
bitstream loaded with the core not processing -- the clock tree toggles in both
phases and cancels with everything else. The dynamic figure is therefore
switching energy of the datapath and the PS work around it, which is what a
duty-cycled device actually spends, and is slightly smaller than a
clock-gated-vs-running difference would be.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:          # not POSIX
    # The I2C side needs the board. The arithmetic below does not, and keeping
    # it importable anywhere is what lets the differential logic -- the part
    # that decides whether a reading may be reported at all -- be tested off
    # the board.
    fcntl = None

I2C_SLAVE = 0x0703
INA260_ADDR = 0x40
REG_CURRENT, REG_BUSV, REG_POWER = 0x01, 0x02, 0x03
REG_MFG_ID, REG_DIE_ID = 0xFE, 0xFF
MFG_TI, DIE_INA260 = 0x5449, 0x2270          # "TI", and the INA260 die
LSB_CURRENT_A, LSB_BUSV_V, LSB_POWER_W = 1.25e-3, 1.25e-3, 10e-3


class INA260:
    """Direct I2C, because xlnx_platformstats looks for a hwmon node K26 has not."""

    def __init__(self, bus: int, addr: int = INA260_ADDR) -> None:
        if fcntl is None:
            raise SystemExit("this needs a POSIX host with /dev/i2c-*; run it on "
                             "the KV260, not on a workstation.")
        self.path = f"/dev/i2c-{bus}"
        self.fd = os.open(self.path, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def read_u16(self, reg: int) -> int:
        os.write(self.fd, bytes([reg]))
        hi, lo = os.read(self.fd, 2)
        return (hi << 8) | lo

    def read_s16(self, reg: int) -> int:
        v = self.read_u16(reg)
        return v - 0x10000 if v & 0x8000 else v

    def identify(self) -> tuple[int, int]:
        return self.read_u16(REG_MFG_ID), self.read_u16(REG_DIE_ID)

    def sample(self) -> dict:
        # Both the power register and V*I. The power register's LSB is 10 mW,
        # which is coarse against the deltas expected here; V*I resolves finer
        # and the two are reported side by side so a disagreement is visible
        # rather than averaged away.
        i = self.read_s16(REG_CURRENT) * LSB_CURRENT_A
        v = self.read_u16(REG_BUSV) * LSB_BUSV_V
        p = self.read_u16(REG_POWER) * LSB_POWER_W
        return {"t": time.monotonic(), "v": v, "i": i, "p_reg": p, "p_vi": v * i}

    def close(self) -> None:
        os.close(self.fd)


def find_bus(explicit: int | None) -> int:
    """The bus with an INA260 answering at 0x40, confirmed by its IDs.

    Guessing is not safe: 0x40 is a common address and answering it does not
    make a device a power monitor. The manufacturer and die IDs do.
    """
    candidates = [explicit] if explicit is not None else sorted(
        int(n.split("-")[1]) for n in os.listdir("/dev")
        if n.startswith("i2c-") and n.split("-")[1].isdigit())
    tried = []
    for bus in candidates:
        try:
            dev = INA260(bus)
        except OSError as e:
            tried.append(f"i2c-{bus}: {e.strerror}")
            continue
        try:
            mfg, die = dev.identify()
            if (mfg, die) == (MFG_TI, DIE_INA260):
                dev.close()
                return bus
            tried.append(f"i2c-{bus}: answered 0x40 with mfg=0x{mfg:04X} die=0x{die:04X}")
        except OSError as e:
            tried.append(f"i2c-{bus}: 0x40 did not respond ({e.strerror})")
        finally:
            try:
                dev.close()
            except OSError:
                pass
    raise SystemExit(
        "no INA260 found at 0x40 on any i2c bus.\n  " + "\n  ".join(tried) +
        "\n\nThe KV260 carries one on VCC_SOM (UG1089). If none answers, check that "
        "i2c-dev is\nloaded (`modprobe i2c-dev`) and that this is running as root.")


def sample_for(dev: INA260, seconds: float, interval: float, label: str) -> list[dict]:
    print(f"  sampling {label} for {seconds:.0f} s ...", flush=True)
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        out.append(dev.sample())
        time.sleep(interval)
    return out


def summarise(rows: list[dict], key: str) -> dict:
    xs = [r[key] for r in rows]
    return {"n": len(xs), "mean": st.mean(xs), "sd": st.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs)}


def differential(before: list[dict], after: list[dict], active: list[dict],
                 key: str, wall_s: float, inferences: int | None,
                 macs: int) -> dict:
    """Active minus idle, with the floor that difference has to clear.

    The floor is the largest of the two baselines' spreads and the drift
    between them. Drift belongs in it: a long run warms the board, and a
    rising idle reads as rising dynamic power to anyone who did not measure
    the idle twice.
    """
    b, f, act = (summarise(before, key), summarise(after, key), summarise(active, key))
    idle_mean = (b["mean"] + f["mean"]) / 2.0
    drift = f["mean"] - b["mean"]
    noise = max(b["sd"], f["sd"], abs(drift))
    delta = act["mean"] - idle_mean
    out = {"idle_W": idle_mean, "active_W": act["mean"], "delta_W": delta,
           "baseline_drift_W": drift, "noise_floor_W": noise,
           "delta_over_noise": (delta / noise) if noise > 0 else None,
           "credible": bool(noise > 0 and delta > 3.0 * noise)}
    if inferences:
        per = delta * wall_s / inferences
        out["energy_per_inference_J"] = per
        out["energy_per_inference_uJ"] = per * 1e6
        out["pJ_per_MAC"] = per / macs * 1e12
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where the readings and the summary go")
    ap.add_argument("--bus", type=int, default=None, help="i2c bus; discovered when omitted")
    ap.add_argument("--baseline-s", type=float, default=60.0)
    ap.add_argument("--interval-s", type=float, default=0.05)
    ap.add_argument("--inferences", type=int, default=None,
                    help="how many inferences the workload performs; without it "
                         "only powers are reported, not energy per inference")
    ap.add_argument("--macs", type=int, default=489_472,
                    help="MACs executed on the PL per inference (13 layers; GAP "
                         "and FC run on the PS)")
    ap.add_argument("workload", nargs=argparse.REMAINDER,
                    help="after `--`, the command that runs the inferences")
    a = ap.parse_args()

    if fcntl is None or not hasattr(os, "geteuid"):
        raise SystemExit(
            "this measures a board: run it on the KV260 over ssh, as root. "
            "A workstation has no /dev/i2c-* and no INA260 to read.")
    if os.geteuid() != 0:
        raise SystemExit("the i2c device needs root: run with sudo")
    work = a.workload[1:] if a.workload and a.workload[0] == "--" else a.workload
    if not work:
        raise SystemExit("give the workload command after `--`")

    bus = find_bus(a.bus)
    dev = INA260(bus)
    mfg, die = dev.identify()
    print(f"INA260 on i2c-{bus} at 0x{INA260_ADDR:02X} "
          f"(mfg 0x{mfg:04X}, die 0x{die:04X})  rail: VCC_SOM, whole module\n")

    before = sample_for(dev, a.baseline_s, a.interval_s, "idle (before)")
    print(f"  running workload: {' '.join(work[:3])} ...", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(work)
    active = []
    while proc.poll() is None:
        active.append(dev.sample())
        time.sleep(a.interval_s)
    wall = time.monotonic() - t0
    rc = proc.returncode
    print(f"  workload finished in {wall:.1f} s, exit {rc}", flush=True)
    after = sample_for(dev, a.baseline_s, a.interval_s, "idle (after)")
    dev.close()

    if rc != 0:
        raise SystemExit(f"the workload exited {rc}; measuring a failed run measures "
                         "nothing. Fix the run, then measure it.")
    if not active:
        raise SystemExit("the workload finished before a single sample was taken; "
                         "lengthen it or shorten --interval-s.")

    res = {"i2c_bus": bus, "rail": "VCC_SOM (whole SOM: PS + PL)",
           "workload": work, "workload_wall_s": wall, "interval_s": a.interval_s,
           "idle_before": {k: summarise(before, k) for k in ("p_reg", "p_vi", "v", "i")},
           "idle_after": {k: summarise(after, k) for k in ("p_reg", "p_vi", "v", "i")},
           "active": {k: summarise(active, k) for k in ("p_reg", "p_vi", "v", "i")}}

    # Drift between the two baselines is the floor the delta has to clear. A
    # long run warms the board, and a rising idle reads as rising dynamic
    # power if it is not stated.
    for key in ("p_reg", "p_vi"):
        res[f"differential_{key}"] = differential(
            before, after, active, key, wall, a.inferences, a.macs)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({**res, "samples": {"idle_before": before, "active": active,
                                      "idle_after": after}}, fh, indent=1)

    print(f"\n{'':<24}{'idle':>10}{'active':>10}{'delta':>10}{'noise':>10}")
    for key, label in (("p_vi", "V x I"), ("p_reg", "power register")):
        d = res[f"differential_{key}"]
        print(f"  {label:<22}{d['idle_W']:>9.3f}W{d['active_W']:>9.3f}W"
              f"{d['delta_W']:>9.3f}W{d['noise_floor_W']:>9.3f}W")
    print(f"\n  baseline drift over the run: "
          f"{res['differential_p_vi']['baseline_drift_W']:+.3f} W")

    d = res["differential_p_vi"]
    print()
    if not d["credible"]:
        print("NOT A MEASUREMENT. The difference between idle and active is "
              f"{d['delta_W']:+.3f} W\n  against a noise floor of "
              f"{d['noise_floor_W']:.3f} W. Report nothing from this run.\n"
              "  Lengthen the workload, or raise --baseline-s so the floor is better "
              "known.")
    else:
        print(f"Dynamic power while running: {d['delta_W']:.3f} W "
              f"({d['delta_over_noise']:.1f}x the noise floor).")
        if a.inferences:
            print(f"  {a.inferences} inferences in {wall:.1f} s -> "
                  f"{d['energy_per_inference_uJ']:.1f} uJ per inference, "
                  f"{d['pJ_per_MAC']:.1f} pJ/MAC.")
        print("  This is WHOLE-SOM dynamic energy and includes the PS work around "
              "each\n  inference -- DMA, polling, readback, and GAP and FC on the "
              "CPU. It is not\n  PL-only accelerator energy, and this board cannot "
              "produce one.")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
