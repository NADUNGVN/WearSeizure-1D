"""Draw one board run of a complete recording against the software model.

    python scripts/plot_board_run.py --board-run RUN.json --out fig_board_chb15_40

Writes `<out>.pdf` (vector, for the paper) and `<out>.png` (to look at), sized
for one IEEE column (88 mm). Three panels, all from the same replay that
`hardware_verification_table.py --board-run` scores:

(a) The whole hour: the smoothed seizure probability the device alarms on,
    computed once from the golden model and once from the board's own feature
    maps, with the annotated seizures shaded and every alarm marked.
(b) The three seizures up close: onset, alarm, delay.
(c) For every window, how many of the 2,048 feature values the board returned
    differ from the golden model.

Nothing here is taken from the run's own claims: the golden model is replayed,
the board's bytes are compared value by value, and its alarms are recomputed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("hvt", REPO / "scripts" / "hardware_verification_table.py")
hvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hvt)
sys.path.insert(0, str(REPO / "src"))

from wearseizure.postprocess.hysteresis import PostprocessParams, hysteresis_alarms

# Okabe-Ito
SOFTWARE, BOARD, SEIZURE, ALARM, GREY = "#56B4E9", "#000000", "#E69F00", "#D55E00", "#999999"


def replay(run: dict, model_dir: Path, vectors: Path) -> dict:
    """Golden and board scores per window, and the alarms each would raise."""
    contract = json.loads((vectors / "contract.json").read_text(encoding="utf-8"))
    if run.get("fold_id") != contract["provenance"]["fold_id"]:
        raise SystemExit(f"board run is fold {run.get('fold_id')}, vectors are {contract['provenance']['fold_id']}")
    if not run.get("complete"):
        raise SystemExit("board run is incomplete; plot only complete recordings")
    with (vectors / "windows.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    board_by_idx = {int(w["window_idx"]): w["features_hex"] for w in run["windows"]}
    if sorted(board_by_idx) != [int(r["window_idx"]) for r in rows]:
        raise SystemExit("board run does not cover exactly the recording's windows")

    stream = hvt.verify_onset_vectors.read_stream(vectors / "stream_input.hex", 8)
    golden_mod, golden, _ = hvt.load_reference_model(model_dir)
    pl_last = next(k for k in golden.specs if k.type == "gap").layer_id - 1
    n_diff, m_gold, m_board = [], [], []
    for r in rows:
        off = int(r["stream_offset_samples"])
        logits, trace = golden.run(stream[off:off + 1024])
        ref = next(v for k, v in trace.items() if k.startswith(f"{pl_last:02d}_"))
        board = np.frombuffer(bytes.fromhex(board_by_idx[int(r["window_idx"])]), dtype=np.int8)
        board = board.astype(np.int64).reshape(ref.shape)
        n_diff.append(int(np.count_nonzero(board != ref)))
        m_gold.append(int(logits[1, 0]) - int(logits[0, 0]))
        l0, l1 = hvt.host_logits(golden_mod, golden, board)
        m_board.append(l1 - l0)

    pp = contract["postprocess"]
    p_out = int(contract["scoring"]["logit_exponent"])
    params = PostprocessParams(
        method=pp["method"], ema_alpha=float(pp["ema_alpha"]), threshold_on=float(pp["threshold_on"]),
        threshold_off=float(pp["threshold_off"]), run_length=int(pp["run_length"]),
        event_merge_gap_s=float(pp["event_merge_gap_s"]), alarm_timestamp=pp["alarm_timestamp"],
        window_s=float(pp["window_s"]),
    )
    end_sec = [float(r["end_sec"]) for r in rows]

    def ema(margins):
        out, e = [], float(pp["ema_warm_start_probability"])
        for m in margins:
            e = params.ema_alpha * (1.0 / (1.0 + np.exp(-m / 2.0 ** p_out))) + (1 - params.ema_alpha) * e
            out.append(e)
        return np.array(out)

    s_gold, s_board = ema(m_gold), ema(m_board)
    stride = end_sec[1] - end_sec[0]
    return {
        "recording": contract["recording"], "t": np.array(end_sec), "n_diff": np.array(n_diff),
        "values_per_window": int(ref.size), "gold": s_gold, "board": s_board, "params": params,
        "seizures": hvt.seizures_from_labels(rows, params.window_s, stride),
        "alarms_gold": hysteresis_alarms(end_sec, list(s_gold), params),
        "alarms_board": hysteresis_alarms(end_sec, list(s_board), params),
    }


def draw(d: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.5, "lines.linewidth": 0.8, "pdf.fonttype": 42,
    })
    p, t_min = d["params"], d["t"] / 60.0
    fig = plt.figure(figsize=(88 / 25.4, 95 / 25.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.45], hspace=0.75, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    for on, off in d["seizures"]:
        ax.axvspan(on / 60, off / 60, color=SEIZURE, alpha=0.35, lw=0)
    ax.plot(t_min, d["gold"], color=SOFTWARE, lw=2.2, label="Software model")
    ax.plot(t_min, d["board"], color=BOARD, lw=0.6, ls=(0, (2, 1.5)), label="KV260 board")
    ax.axhline(p.threshold_on, color=GREY, lw=0.5, ls="--")
    ax.text(0.5, p.threshold_on + 0.03, r"$\theta_{on}$", ha="left", va="bottom", fontsize=6.5)
    for a in d["alarms_board"]:
        ax.plot(a[0] / 60, 1.1, marker="v", color=ALARM, ms=4)
    ax.set_xlim(0, t_min[-1])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Smoothed seizure\nprobability")
    ax.set_title(f"(a) Whole recording {d['recording']}: {len(d['alarms_board'])} alarms (▼), "
                 f"{len(d['seizures'])} seizures", loc="left")
    handles = ax.get_legend_handles_labels()[0] + [plt.Rectangle((0, 0), 1, 1, color=SEIZURE, alpha=0.35)]
    ax.legend(handles, ["Software model", "KV260 board", "Annotated seizure"], loc="upper left",
              bbox_to_anchor=(0.27, 0.97), frameon=False, ncol=1, handlelength=1.8)

    for i, (on, off) in enumerate(d["seizures"]):
        ax = fig.add_subplot(gs[1, i])
        sel = (d["t"] >= on - 60) & (d["t"] <= on + 90)
        rel = d["t"][sel] - on
        ax.axvspan(0, min(off - on, 90), color=SEIZURE, alpha=0.35, lw=0)
        ax.plot(rel, d["gold"][sel], color=SOFTWARE, lw=2.2)
        ax.plot(rel, d["board"][sel], color=BOARD, lw=0.6, ls=(0, (2, 1.5)))
        ax.axhline(p.threshold_on, color=GREY, lw=0.5, ls="--")
        hit = [a for a in d["alarms_board"] if on <= a[0] <= off]
        if hit:
            delay = hit[0][0] - on
            ax.plot(delay, 1.1, marker="v", color=ALARM, ms=4)
            ax.annotate(f"{delay:.0f} s", (delay + 7, 1.1), ha="left", va="center", fontsize=6.5)
        ax.set_xlim(-60, 90)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([-60, 0, 60])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Probability")
        ax.set_title(f"(b) Seizure {i + 1}" if i == 0 else f"Seizure {i + 1}", fontsize=7,
                     loc="left" if i == 0 else "center")
        if i == 1:
            ax.set_xlabel("Time from annotated onset (s)")

    ax = fig.add_subplot(gs[2, :])
    total = d["n_diff"].size * d["values_per_window"]
    ax.plot(t_min, d["n_diff"], color=BOARD, lw=0.8)
    ax.set_xlim(0, t_min[-1])
    ax.set_ylim(-0.5, 5)
    ax.set_yticks([0, 5])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Differing\nvalues")
    ax.set_title(f"(c) Board vs. golden feature maps: {int(d['n_diff'].sum()):,} of {total:,} values differ",
                 loc="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--board-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = ap.parse_args()

    run = json.loads(args.board_run.read_text(encoding="utf-8"))
    match = [(m, v) for m, v in hvt.whole_recordings(REPO)
             if json.loads((v / "contract.json").read_text(encoding="utf-8"))["recording"] == run["recording"]]
    if not match:
        raise SystemExit(f"no exported complete recording named {run['recording']}")
    print(f"replaying {run['recording']} ({len(run['windows'])} windows) ...", flush=True)
    d = replay(run, *match[0])
    same = [list(a) for a in d["alarms_board"]] == [list(a) for a in d["alarms_gold"]]
    print(f"differing values: {int(d['n_diff'].sum())}; alarms identical to software: {same}")
    draw(d, args.out)
    print(f"written {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")
    return 0 if same and not d["n_diff"].any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
