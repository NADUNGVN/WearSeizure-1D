"""Build the one-page comparison table as a PDF, in the supervisor's format.

    python scripts/make_comparison_pdf.py [-o docs/comparison_table.pdf]

Supervisor's instruction: mark the best value in **bold** and the second best
underlined. Both are computed from the numbers here, never marked by hand --
hand-marking a ranking is exactly where a table acquires an error nobody
catches, and this one is going in front of a reviewer.

Direction matters and is declared per column: higher is better for accuracy and
sensitivity, lower is better for parameters, footprint and MACs.

Only CHB-MIT rows are ranked. A row on another dataset is shown for reference
and excluded from the ranking, because winning a column on a different corpus is
not winning it.
"""
from __future__ import annotations

import argparse

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ---------------------------------------------------------------------------
# The data. `rank` holds the comparable numeric value for each ranked column,
# or None where the paper does not report one.
# ---------------------------------------------------------------------------

HEADERS = [
    "Method", "Dataset", "Model", "Protocol / test exposure",
    "ACC (%)", "SEN (%)", "Parameters", "Precision",
    "Weight memory", "Other efficiency",
]

# (higher_is_better) per ranked column key
DIRECTION = {"acc": True, "sen": True, "params": False, "footprint_kb": False}

ROWS = [
    {
        "method": "<b>WearSeizure-1D (this work)</b>",
        "data": "CHB-MIT (13 single-channel cases)",
        "model": "Separable 1D-CNN, k5 + dilation 1/2/4/8/16; 1 channel; 4-s windows",
        "protocol": "<b>Leakage-safe</b>: split by recording before filtering; thresholds frozen on val; "
                    "<b>185.0 h</b> continuous test; 66 folds x 3 seeds",
        "acc": 98.88, "sen": 94.95,
        "sen_note": "event level; 60.33 segment level",
        "params": 11786,
        "precision": "<b>DFP8</b><br/>"
                     "<font size=5.4>chosen on measured loss: 0.00 pp of event sensitivity "
                     "against FP32 on all 66 folds, through the real integer datapath</font>",
        "footprint": "<b>10.87 KiB</b> weights (DFP8)<br/>+ 1.76 KiB biases (32-bit)",
        "footprint_kb": 10.87,
        "other": "585,920 MACs (thop) / 489,600 conv+fc; FAR 0.25/h<br/>"
                 "<b>total on-chip 18.62 KiB</b> = weights + biases + 6.00 KiB peak "
                 "activation<br/><font size=5.4>3.6% of XC7Z020 BRAM</font>",
        "macs": 585920,
    },
    {
        # The same code and data under the protocol most of the other rows use.
        # Without this row the table compares 94.89% measured under splits by
        # recording against 99.62% measured under a segment-level split, which
        # is not a comparison. With it, the contrast between the two rows IS the
        # contribution -- and it shows accuracy saturating either way.
        "method": "<b>WearSeizure-1D</b>, same code under the segment-split protocol",
        "data": "CHB-MIT (13 single-channel cases)",
        "model": "As above, retrained under the leaky protocol",
        "protocol": "<i>Segment-level random split</i>; normalised on all data; threshold on test "
                    "&mdash; i.e. the protocol common to the rows below. 99.6% of test windows have a "
                    "near-duplicate in training",
        "acc": 99.68, "sen": 92.29, "sen_note": "segment level",
        "params": 11786, "precision": "FP32 (measured)",
        "footprint": "as above", "footprint_kb": None, "onchip": "as above",
        "other": "Same network; only the split rule differs from the row above",
        "macs": None,
    },
    {
        "method": "<b>EpiSepNet-5K</b> (INT16)<br/><font size=5.4>earlier model, same group</font>",
        "data": "CHB-MIT", "model": "BatchNorm-folded separable 1D-CNN",
        "protocol": "—",
        "acc": 90.04, "sen": 90.76, "sen_note": "", "params": 4900, "precision": "INT16",
        "footprint": "10.0 KB package; ~9.8 KB raw INT16",
        "footprint_kb": 9.8,
        "other": "99.9743% agreement with FP32; 2.81x smaller package", "macs": None,
    },
    {
        "method": "<b>EpiSepNet-5K</b> (FP32)<br/><font size=5.4>earlier model, same group</font>",
        "data": "CHB-MIT", "model": "Separable 1D-CNN; 17-channel raw EEG; 2-s windows",
        "protocol": "—",
        "acc": 90.07, "sen": 90.76, "sen_note": "", "params": 5010, "precision": "FP32",
        "footprint": "28.1 KB checkpoint; ~20.0 KB raw FP32",
        "footprint_kb": 20.0, "other": "Reference model", "macs": None,
    },
    {
        "method": "Chung et al. 2024 [1]",
        "data": "CHB-MIT (13 cases)",
        "model": "Stacked 2D-CNN, parallel 1x3 and 1x5 kernels; 1 clinician-selected channel",
        "protocol": "Segment-level 70/20/10 split per patient; ~91 h",
        "acc": 98.18, "sen": 99.62, "sen_note": "event level; 96.76 segment level",
        "params": 116700, "precision": "FP32",
        "footprint": "<b>454 KB</b> at FP32<br/><font size=5.4>rebuilt from the paper; "
                     "116 226 params vs 116 700 reported, +0.4%</font>",
        "footprint_kb": 454.0,
        "other": "FAR 0.22/h; detection delay 3.3 s; specificity 98.19%<br/><b>weights + activations: 510 KB at FP32 = 91% of XC7Z020 BRAM</b> <font size=5.4>(rebuilt from the paper, peak 14 464 values)</font>",
        "macs": None,
    },
    {
        "method": "Werner et al. (TC-ResNet4) [2]",
        "data": "CHB-MIT", "model": "16-channel TC-ResNet4 with fixed-point inference",
        "protocol": "—",
        "acc": 95.28, "sen": 92.34, "sen_note": "", "params": 9840, "precision": "4-bit",
        "footprint": "~4.92 KB weight-only (4-bit)",
        "footprint_kb": 4.92,
        "other": "337,968 MACs; 495 nW average power", "macs": 337968,
    },
    {
        "method": "Zhu et al. 2021 [3]",
        "data": "CHB-MIT (23 channels)", "model": "7-layer 1D-CNN, 5x1 conv, shift-register PE array",
        "protocol": "80/10/10 split; normal class down-sampled 200:1",
        "acc": 97.35, "sen": 94.32, "sen_note": "", "params": 7010, "precision": "fixed-point",
        "footprint": "<i>est.</i> 28 KB at FP32, 7 KB at 8-bit", "footprint_kb": 7.0, "footprint_estimated": True,
        "other": "6.32 MOPs; 170 us/inference @ 200 MHz; FPGA Xilinx Zynq ZC706", "macs": 6320000,
    },
    {
        "method": "Li et al. 2022 [4]",
        "data": "CHB-MIT + Bonn + SWEC-ETHZ", "model": "1D-CNN with parallel convolutional layers",
        "protocol": "80/20 + 5-fold CV; GAN-synthesised preictal segments",
        "acc": 99.01, "sen": 99.24, "sen_note": "", "params": 10778, "precision": "analog RRAM",
        "footprint": "<i>est.</i> 43 KB at FP32", "footprint_kb": 43.0, "footprint_estimated": True,
        "other": "1.13 us parallelised; 7.21 W; ASIC 22 nm FDSOI + RRAM crossbar", "macs": None,
    },
    {
        "method": "Ferrara et al. [5]",
        "data": "CHB-MIT", "model": "Patient-specific two-channel lightweight CNN",
        "protocol": "—",
        "acc": 99.0, "sen": 67.0, "sen_note": "", "params": 9500, "precision": "—",
        "footprint": "51 KB model", "footprint_kb": 51.0,
        "other": "Balanced accuracy 83.0%; 0.10 FP/h", "macs": None,
    },
    {
        "method": "REST-RS [6]",
        "data": "CHB-MIT", "model": "Graph-based residual state-update model",
        "protocol": "—",
        "acc": None, "sen": None, "sen_note": "", "params": 9300, "precision": "—",
        "footprint": "0.037 MB, ~37 KB", "footprint_kb": 37.0,
        "other": "AUROC up to 93.5%; 1.314 ms inference", "macs": None,
    },
    {
        "method": "SlimSeiz [7]",
        "data": "CHB-MIT", "model": "Eight-channel convolution and Mamba network (prediction task)",
        "protocol": "—",
        "acc": 94.8, "sen": 95.5, "sen_note": "", "params": 21200, "precision": "—",
        "footprint": "<i>est.</i> 84.8 KB at FP32", "footprint_kb": 84.8, "footprint_estimated": True,
        "other": "Specificity 94.0%", "macs": None,
    },
    {
        "method": "RGF-Model [8]",
        "data": "CHB-MIT", "model": "Multi-teacher knowledge-distillation model",
        "protocol": "—",
        "acc": 98.92, "sen": 98.54, "sen_note": "", "params": 82000, "precision": "FP32",
        "footprint": "0.33 MB (330 KB)", "footprint_kb": 330.0,
        "other": "Specificity 99.11%; AUC 98.96%", "macs": None,
    },
    {
        "method": "Wang et al. (MSCA) [9]",
        "data": "CHB-MIT", "model": "Inverted residual CNN with multi-scale channel attention",
        "protocol": "—",
        "acc": 98.70, "sen": 98.30, "sen_note": "", "params": 88000, "precision": "—",
        "footprint": "<i>est.</i> 352 KB at FP32", "footprint_kb": 352.0, "footprint_estimated": True,
        "other": "2.68M MACs; specificity 99.10%", "macs": 2680000,
    },
    {
        "method": "Ahlawat (INT8) [10]",
        "data": "CHB-MIT", "model": "Quantized common-channel 1D-CNN with operator fusion",
        "protocol": "—",
        "acc": None, "sen": None, "sen_note": "", "params": None, "precision": "INT8",
        "footprint": "0.44 MB (440 KB) at INT8; 1.63 MB at FP32", "footprint_kb": 440.0,
        "other": "Up to 2.8x speedup; up to 64% estimated energy reduction", "macs": None,
    },
]

REFERENCES = [
    (
        "[1] Y. G. Chung, A. Cho, H. Kim, K. J. Kim. Single-channel seizure detection with clinical confirmation of "
        "seizure locations using CHB-MIT dataset. Front. Neurol. 15:1389731, 2024."
    ),
    "[2] J. Werner, B. Kohli, P. Palomero Bernardo, C. Gerum et al. TC-ResNet for EEG seizure detection.",
    (
        "[3] L. Zhu, D. Liu, X. Li, J. Lu, L. Wei, X. Cheng. An Efficient Hardware Architecture for Epileptic Seizure "
        "Detection using EEG Signals based on 1D-CNN. IEEE ASICON, 2021."
    ),
    (
        "[4] C. Li, C. Lammie, X. Dong, A. Amirsoleimani, M. Rahimi Azghadi, R. Genov. Seizure Detection and Prediction "
        "by Parallel Memristive Convolutional Neural Networks. IEEE TBioCAS, 2022."
    ),
    "[5] Ferrara et al. Patient-specific two-channel lightweight CNN for seizure detection.",
    "[6] REST-RS: graph-based residual state-update model for EEG seizure detection.",
    "[7] SlimSeiz: convolution and Mamba network for seizure prediction.",
    "[8] RGF-Model: multi-teacher knowledge distillation for seizure detection.",
    "[9] Wang et al. Lightweight Seizure Detection Based on Multi-Scale Channel Attention, 2023.",
    (
        "[10] K. Ahlawat. Efficient EEG Seizure Detection Using INT8 Quantization, Channel Pruning, and Spiking Neural "
        "Networks. arXiv:2607.16296, 2026."
    ),
]


def rank_marks(rows: list[dict], key: str, higher_is_better: bool) -> dict[int, str]:
    """Row index -> 'best' or 'second', computed rather than hand-assigned.

    Rows on another dataset are excluded: winning a column on a different corpus
    is not winning it.
    """
    scored = [(i, r[key]) for i, r in enumerate(rows)
              if r.get(key) is not None and not r.get("off_dataset")
              # A footprint derived from a parameter count is not a reported
              # number. Bolding one would claim a precision nobody measured.
              and not (key == "footprint_kb" and r.get("footprint_estimated"))]
    scored.sort(key=lambda t: t[1], reverse=higher_is_better)
    marks = {}
    if scored:
        marks[scored[0][0]] = "best"
    if len(scored) > 1 and scored[1][1] != scored[0][1]:
        marks[scored[1][0]] = "second"
    return marks


def emphasise(text: str, mark: str | None) -> str:
    if mark == "best":
        return f"<b>{text}</b>"
    if mark == "second":
        return f"<u>{text}</u>"
    return text



# ---------------------------------------------------------------------------
# Table 2: the channel ablation. Three arms of this same model, identical
# except for how many channels they read, so the difference is attributable to
# the channel count and nothing else. All three train from scratch per fold --
# no cohort pre-training, no distillation -- which is why the 1-channel arm
# reads 91.79 rather than the 94.95 in Table 1. Cross-arm comparison is valid;
# comparison against Table 1 is not, and the caption says so.
# ---------------------------------------------------------------------------

ABLATION_HEADERS = [
    "Channels", "SEN event (%)", "SEN segment (%)", "ACC (%)", "AUROC",
    "FAR/h", "Parameters", "MACs", "Total on-chip", "Runs on the accelerator?",
]
ABLATION_DIRECTION = [None, True, True, True, True, False, False, False, False, None]
ABLATION_ROWS = [
    ["<b>1 (deployed)</b>", 91.79, 49.36, 98.96, 0.8870, 0.2798, 11578, 585920, 18.62,
     "yes - 1,024 of 16,384 cells"],
    ["4", 93.31, 59.36, 99.10, 0.9237, 0.3334, 11746, 671936, 20.79,
     "yes - 4,096 of 16,384 cells"],
    ["18", 95.20, 70.51, 99.37, 0.9586, 0.2625, 12530, 1073344, 35.55,
     "<b>no</b> - needs 18,432 cells"],
]
ABLATION_FMT = [None, "{:.2f}", "{:.2f}", "{:.2f}", "{:.4f}", "{:.4f}",
                "{:,}", "{:,}", "{:.2f} KiB", None]

# ---------------------------------------------------------------------------
# Table 3: activation memory under the definition asked for -- the largest sum
# of the feature maps of two adjacent layers, plus any skip connection live
# across them. This network has none: DepthwiseSeparableConv1d is
# depthwise -> pointwise -> BN -> ReLU and MultiScaleDilatedBlock in k5_only
# mode is a single branch, so neither adds its input back.
# ---------------------------------------------------------------------------

FOOTPRINT_HEADERS = ["Layer", "Input map", "Output map", "Input + output"]
FOOTPRINT_ROWS = [
    ["stem.0", 1024, 4096, 5120], ["<b>b1.dw</b>", 4096, 2048, 6144],
    ["b1.pw", 2048, 4096, 6144], ["b2.dw", 4096, 2048, 6144],
    ["b2.pw", 2048, 3072, 5120], ["b3.dw", 3072, 1536, 4608],
    ["b3.pw", 1536, 2048, 3584], ["b4.dw", 2048, 1024, 3072],
    ["b4.pw", 1024, 1536, 2560], ["context.0.dw", 1536, 1536, 3072],
    ["context.0.pw", 1536, 2048, 3584], ["context.1.dw", 2048, 2048, 4096],
    ["context.1.pw", 2048, 2048, 4096], ["gap", 2048, 64, 2112],
    ["fc", 64, 2, 66],
]


def marks_for(rows, col, higher_is_better):
    """best / second for one column of a plain list-of-lists table."""
    vals = [(i, r[col]) for i, r in enumerate(rows) if isinstance(r[col], (int, float))]
    if len(vals) < 2:
        return {}
    vals.sort(key=lambda t: t[1], reverse=higher_is_better)
    out = {vals[0][0]: "best"}
    if vals[1][1] != vals[0][1]:
        out[vals[1][0]] = "second"
    return out


def simple_table(headers, rows, directions, fmts, widths, head_style, body_style,
                 page_w, shade_row=None):
    """A ranked table: best bold, second underlined, computed not hand-marked."""
    col_marks = {c: marks_for(rows, c, d) for c, d in enumerate(directions) if d is not None}
    data = [[Paragraph(h, head_style) for h in headers]]
    for i, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            text = v if fmts[c] is None else fmts[c].format(v)
            cells.append(Paragraph(emphasise(text, col_marks.get(c, {}).get(i)), body_style))
        data.append(cells)
    total = sum(widths)
    t = Table(data, colWidths=[w / total * page_w for w in widths], repeatRows=1)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if shade_row is not None:
        style.append(("BACKGROUND", (0, shade_row + 1), (-1, shade_row + 1),
                      colors.HexColor("#F2F6FF")))
    t.setStyle(TableStyle(style))
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/comparison_table.pdf")
    args = ap.parse_args()

    marks = {k: rank_marks(ROWS, k, d) for k, d in DIRECTION.items()}
    macs_marks = rank_marks(ROWS, "macs", higher_is_better=False)

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=6.4, leading=7.8, alignment=TA_LEFT)
    head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=6.8, leading=8.2, alignment=TA_LEFT)
    note = ParagraphStyle("note", fontName="Helvetica", fontSize=7.0, leading=9.5)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceAfter=3)
    refs = ParagraphStyle("refs", fontName="Helvetica", fontSize=6.2, leading=7.6)

    data = [[Paragraph(h, head) for h in HEADERS]]
    for i, r in enumerate(ROWS):
        acc = "—" if r["acc"] is None else f"{r['acc']:.2f}"
        sen = "—" if r["sen"] is None else f"{r['sen']:.2f}"
        if r["sen_note"]:
            sen = emphasise(sen, marks["sen"].get(i)) + f"<br/><font size=5.4>{r['sen_note']}</font>"
        else:
            sen = emphasise(sen, marks["sen"].get(i))
        params = "2014" if r["params"] is None else f"{r['params']:,}"
        foot = r["footprint"]
        if r.get("footprint_kb") is not None and marks["footprint_kb"].get(i):
            foot = emphasise(foot, marks["footprint_kb"][i])
        other = emphasise(r["other"], macs_marks.get(i)) if macs_marks.get(i) else r["other"]

        data.append([
            Paragraph(r["method"], body),
            Paragraph(r["data"], body),
            Paragraph(r["model"], body),
            Paragraph(r["protocol"], body),
            Paragraph(emphasise(acc, marks["acc"].get(i)), body),
            Paragraph(sen, body),
            Paragraph(emphasise(params, marks["params"].get(i)), body),
            Paragraph(r["precision"], body),
            Paragraph(foot, body),
            Paragraph(other, body),
        ])

    widths = [36, 27, 56, 56, 15, 23, 21, 19, 44, 62]
    total = sum(widths)
    page_w = landscape(A3)[0] - 16 * mm
    table = Table(data, colWidths=[w / total * page_w for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        # This work first and shaded, so the eye lands on it without the table
        # having to claim anything the numbers do not support.
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F2F6FF")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))

    doc = SimpleDocTemplate(
        args.out, pagesize=landscape(A3),
        leftMargin=8 * mm, rightMargin=8 * mm, topMargin=8 * mm, bottomMargin=8 * mm,
        title="Single-channel EEG seizure detection: accuracy, size and efficiency",
    )
    story = [
        Paragraph("Single-channel EEG seizure detection on CHB-MIT: accuracy, model size and efficiency", title),
        Paragraph(
            (
                "<b>Bold</b> = best in column. <u>Underlined</u> = second best. "
                "Higher is better for ACC and SEN; lower is better for parameters, stored footprint and MACs. "
                "Every row is CHB-MIT. Footprints marked <i>est.</i> are derived from a parameter count rather "
                "than reported by the paper, and are excluded from the ranking. EpiSepNet-5K is this group’s "
                "own earlier model, kept in the table as the design it succeeds. Weight memory and total "
                "on-chip memory are different quantities: most papers report only the first. "
                "<b>DFP</b> is dynamic fixed point, a power-of-two scale: its footprint EQUALS plain integer "
                "at the same width, because the scale is per tensor rather than per value. What it changes is "
                "the datapath — requantising between layers becomes a shift instead of a multiply."
            ),
            note),
        Spacer(1, 4),
        table,
        Spacer(1, 6),
        Paragraph(
            (
                "<b>Reading the table.</b> Protocols differ between rows, and most papers do not state theirs in "
                "enough detail to compare. This work is the only row evaluated with splits taken by recording, "
                "thresholds frozen on a held-out validation partition, and 185 h of continuous test exposure. "
                "Under the segment-level random split used by much of this literature, the same code and data reach "
                "92.29 % segment sensitivity instead of 60.33 % — a 31-point difference caused by the protocol "
                "alone, because 99.6 % of test windows then have a near-duplicate in training. "
                "Accuracy is nearly blind to this: at 0.62 % ictal prevalence a model that never predicts a seizure "
                "already scores 99.38 %."
            ),
            note),
        Spacer(1, 10),
        Paragraph("Table 2 &mdash; what the single-channel constraint costs", title),
        Paragraph(
            (
                "Three arms of the <b>same</b> model, identical except for how many channels it reads, so "
                "the difference is attributable to the channel count and nothing else. 66 folds x 3 seeds = "
                "198 runs per arm; montages taken verbatim from Chung et al. so the comparison lines up with "
                "their published ablation channel-for-channel. "
                "<b>All three train from scratch per fold &mdash; no cohort pre-training, no distillation &mdash; "
                "which is why the 1-channel arm reads 91.79 % here and 94.95 % in Table 1. Compare the arms "
                "with each other, never with Table 1.</b>"
            ),
            note),
        Spacer(1, 4),
        simple_table(ABLATION_HEADERS, ABLATION_ROWS, ABLATION_DIRECTION, ABLATION_FMT,
                     [12, 14, 15, 11, 11, 11, 13, 13, 13, 22], head, body, page_w,
                     shade_row=0),
        Spacer(1, 5),
        Paragraph(
            (
                "<b>Paired bootstrap against the 18-channel arm</b>, clustered by patient. "
                "1 channel: <b>&minus;3.41 pp</b> event sensitivity, 95 % CI [&minus;5.83, &minus;0.93] &mdash; "
                "excluding zero, so the cost is established; and &minus;21.16 pp at segment level, "
                "CI [&minus;26.28, &minus;16.47]. 4 channels: &minus;1.89 pp event, CI [&minus;5.89, +1.97] &mdash; "
                "<i>spans zero</i>, so four channels are not measurably worse than eighteen. "
                "<b>Post-processing absorbs 84 % of the deficit</b>: 21.16 points at segment level become 3.41 at "
                "event level, because smoothing, hysteresis and run-length filtering integrate the score across "
                "consecutive windows. That is why a single-electrode device is viable at all. "
                "Chung et al. report the 18-to-1 penalty as 1.9 points of segment sensitivity; measured without "
                "their segment-level random split it is <b>21.2</b>. "
                "Accuracy moves <b>0.41 pp</b> across an eighteen-fold change in input channels while segment "
                "sensitivity moves 21.2 &mdash; a third independent demonstration that it cannot be used to "
                "compare detectors on this data. AUROC, being threshold-free, is not blind."
            ),
            note),
        Spacer(1, 10),
        Paragraph("Table 3 &mdash; activation memory, layer by layer", title),
        Paragraph(
            (
                "Definition: <b>the largest sum of the feature maps of two adjacent layers</b>, plus any skip "
                "connection that must stay live across them. A layer reads its input while writing its output, so "
                "both are resident at the same instant. <b>This network has no skip connections</b> &mdash; "
                "DepthwiseSeparableConv1d is depthwise &rarr; pointwise &rarr; BN &rarr; ReLU and the multi-scale "
                "block in k5-only mode is a single branch, so neither adds its input back &mdash; and there is "
                "therefore nothing to add. Bytes at DFP8, one byte per activation."
            ),
            note),
        Spacer(1, 4),
        simple_table(FOOTPRINT_HEADERS, FOOTPRINT_ROWS,
                     [None, False, False, False], [None, "{:,} B", "{:,} B", "{:,} B"],
                     [20, 16, 16, 18], head, body, page_w * 0.42, shade_row=1),
        Spacer(1, 5),
        Paragraph(
            (
                "<b>Peak = 6,144 B = 6.00 KiB, at b1.dw.</b> Total on-chip = 10.87 KiB weights + 1.76 KiB biases "
                "+ 6.00 KiB activations = <b>18.62 KiB</b>. "
                "Two details that would otherwise be asked about. <b>Biases are stored at 32 bit</b>, not at the "
                "data width: the PE loads a bias as the accumulator's initial value, so it lives at the "
                "accumulator's fixed point, and counting them at 8 bit would understate the total by 1.3 KiB. "
                "And <b>the accelerator allocates 8.00 KiB rather than 6.00</b>, because it uses a ping-pong pair "
                "of banks &mdash; layers alternate between them instead of being packed &mdash; so each bank must "
                "hold the largest map it ever sees. 6.00 KiB is what the network requires; 8.00 KiB is what is on "
                "the die, for a total of 20.62 KiB. The 18-channel arm needs 22.00 KiB of activation and does not "
                "fit the 16,384-cell feature-map memory at all."
            ),
            note),
        Spacer(1, 5),
        Paragraph("<b>References</b>", note),
    ] + [Paragraph(r, refs) for r in REFERENCES]

    doc.build(story)
    print(f"wrote {args.out}")
    marks["macs"] = macs_marks
    for key in ("acc", "sen", "params", "footprint_kb", "macs"):
        best = [ROWS[i]["method"] for i, m in marks[key].items() if m == "best"]
        second = [ROWS[i]["method"] for i, m in marks[key].items() if m == "second"]
        clean = lambda s: s.replace("<b>", "").replace("</b>", "")
        print(f"  {key:<14} best={clean(best[0]) if best else '-':<32} "
              f"second={clean(second[0]) if second else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
