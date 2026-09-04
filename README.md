# WearSeizure-1D

Single-channel EEG seizure **detection** under a leakage-safe protocol, sized for a streaming
INT8 1D-CNN accelerator. Targets IEEE TBioCAS.

CHB-MIT, 13 single-channel-eligible cases, 77 seizures. Evaluated on **185.0 hours** of
continuous held-out recording across 66 folds and 3 seeds, with thresholds frozen on validation
and every split taken **by recording** before any filtering, normalisation or windowing.

Setup and reproduction: **[`docs/RUNNING.md`](docs/RUNNING.md)**.
Frozen model and hardware footprint: **[`docs/MODEL_CARD_k5only.md`](docs/MODEL_CARD_k5only.md)**.

---

## Result

`wearseizure1d_k5only` with cohort pre-training and distillation:

| | |
|---|---:|
| **event sensitivity** (macro / micro) | **0.9489 / 0.9567** |
| false alarms per hour | 0.2937 |
| mean detection delay | 17.75 s |
| worst-patient sensitivity | 0.9333 |
| **parameters / MACs** | **11 786 / 585 920** |
| on-chip SRAM, INT8 | **18.2 KiB** |

---

## 1. Against published work on CHB-MIT

Protocols differ between rows, so these are not like-for-like — section 3 is what makes them
comparable. Sensitivity is **event-level** where the paper reports one, segment-level otherwise.

`W` is weight memory — the number most papers report. `W+A` adds activations, which almost none do.
Footprints marked *est.* are derived as `params x bytes-per-value`, not reported by the paper.

| Work | Ch | Acc | Sens | Params | Precision | W | W+A | MACs |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| Chung 2024, *Front. Neurol.* | **1** | 98.18 % | **99.62 %** (event) / 96.76 % (seg) | 116 700 | FP32 | *est.* 467 KB | — | — |
| Cao 2025, *BMC MIDM* | multi | 98.43 % | 97.84 % | ~5 100 | FP32 | *est.* 20 KB | — | — |
| Hasan 2024, IEEE RAAICON | multi | 98.93 % | 98.60 % | ~4 080 000 | FP32 | *est.* 16.3 MB | — | — |
| Li 2022, *IEEE TBioCAS* | multi | 99.01 % | 99.24 % | 10 778 | RRAM | *est.* 43 KB | in crossbar | — |
| Alharthi 2022, *Sensors* | 18 | 96.87 % | 96.85 % | ~83 300 | FP32 | *est.* 333 KB | — | — |
| Zhu 2021, IEEE ASICON | 23 | 97.35 % | 94.32 % | 7 010 | fixed-pt | *est.* 7 KB | — | 6 320 000 |
| Kashefi Amiri 2025, *Sci. Rep.* | multi | 96.94 % | 92.21 % | 765 000 | FP32 | *est.* 3.06 MB | — | 1 670 000 |
| EpiSepNet-5K | 17 | 90.07 % | 90.76 % | 5 010 | FP32 | 20.0 KB | — | — |
| EpiSepNet-5K | 17 | 90.04 % | 90.76 % | 4 900 | INT16 | **9.8 KB** | — | — |
| Werner et al. (TC-ResNet4) | 16 | 95.28 % | 92.34 % | 9 840 | 4-bit | **4.92 KB** | — | 337 968 |
| Ferrara et al. | 2 | 99.0 % | 67.0 % | ~9 500 | — | 51 KB | — | — |
| SlimSeiz | 8 | 94.8 % | 95.5 % | 21 200 | — | *est.* 84.8 KB | — | — |
| Wang et al. (MSCA) | multi | 98.70 % | 98.30 % | 88 000 | — | *est.* 352 KB | — | 2 680 000 |
| Ahlawat | multi | — | — | — | INT8 | 440 KB | — | — |
| Ali 2024, *R. Soc. Open Sci.* | 18 | — | 75.34 % | — | — | — | — | — |
| **WearSeizure-1D** | **1** | 98.88 % | **94.89 %** (event) / 60.33 % (seg) | **11 786** | INT8 | **11.5 KB** | **18.2 KB** | **585 920** |
| **WearSeizure-1D** | **1** | — | — | **11 786** | DFP16 | 23.0 KB | 36.3 KB | **585 920** |

Two claims this table does **not** support, and which should never be made:

- **Not the smallest model.** 11 786 parameters is more than Cao's ~5 100 and Zhu's 7 010. The
  computational claim is about **MACs**, not parameter count.
- **Part of the advantage is the single channel**, not the architecture. That is the
  contribution, but it has to be said rather than hidden.

What the table *does* support: **one channel, 9.9× fewer parameters than the single-channel
work it is compared against, and the only row evaluated on 185 h without leakage.**

On footprint specifically: 11.5 KB of weights is third behind Werner's 4.92 KB (at 4 bits) and
EpiSepNet's 9.8 KB (at INT16, on a 4 900-parameter model). This work is the only row that reports
**weights + activations**, because it is the only one that had to size an on-chip memory rather
than a stored file. The two are not interchangeable, and comparing 18.2 KB against another paper's
20 KB would be comparing different quantities.

---

## 2. Hardware comparison

The rows that actually built something. Useful reference points for the accelerator work.

| Work | Platform | Params | Memory | Ops | Latency | Power |
|---|---|--:|--:|--:|--:|--:|
| Lee 2024, *IEEE TBioCAS* (RVDLAHA) | Xilinx **PYNQ-Z2** | 356 | *est.* 0.4 KB | 3 909 | 2.1 ms | 107 mW @ 1 MHz |
| Zhu 2021, IEEE ASICON | Xilinx Zynq **ZC706** | 7 010 | *est.* 7 KB | 6.32 MOP | 170 µs @ 200 MHz | 24.96 GOP/s/kLUT |
| Li 2022, *IEEE TBioCAS* | ASIC, 22 nm RRAM crossbar | 10 778 | in RRAM crossbar | — | 1.13 µs | 7.21 W |
| **WearSeizure-1D** (target) | PYNQ-Z2 / Zynq-7020 | **11 786** | **18.2 KB** (W+A) | **489 600 MAC** | ≤ 2 ms budget | not yet measured |

Sizing, derived in the model card: 489 600 MACs inside a 2 ms budget is **245 MMAC/s**, which at
100 MHz is 2.45 MACs/cycle — a **4–8 MAC array**. Total on-chip SRAM is **18.2 KiB** (11.5 KiB
INT8 weights + 6.7 KiB line buffers) — about **3.6 %** of the XC7Z020's BRAM by block count,
not by bytes; see below for why the distinction matters.

One consequence that is easy to get backwards: a decision every second at ~1 ms of compute is a
**0.1 % duty cycle**, so energy per hour is dominated by **static** power. Minimising MACs is not
the energy lever an accelerator project will assume it is.

---

## 2b. Memory footprint

Everything the accelerator must hold on chip, at INT8. Generated by
`python scripts/hardware_spec.py wearseizure1d_k5only`, read off the model rather than transcribed.

The network is the same at every width; only the number of bytes per value changes.

| Format | Weights | Activations, streaming | Activations, layer-sequential | **Total, streaming** | Total, sequential |
|---|--:|--:|--:|--:|--:|
| **FP32** (uncompressed) | 46.0 KiB | 26.7 KiB | 32.2 KiB | **72.7 KiB** | 78.3 KiB |
| INT16 / DFP16 | 23.0 KiB | 13.3 KiB | 16.1 KiB | **36.3 KiB** | 39.1 KiB |
| **INT8 / DFP8** (target) | **11.5 KiB** | **6.7 KiB** | **8.1 KiB** | **18.2 KiB** | 19.6 KiB |

Even **uncompressed FP32 fits**, at about 12 % of the XC7Z020's ~630 KiB of BRAM. Memory is not
what constrains this design, at any width — which is why the numeric format is being chosen by
measuring accuracy loss rather than by counting bytes (`docs/PLAN_quantisation.md`).

Structure of the INT8 column:

| | Bytes | |
|---|--:|---|
| **Weights** | 11 786 B | one byte per parameter; no bias terms, BatchNorm folds into the preceding conv |
| **Line buffers**, fully streaming | 6 823 B | every layer live at once, no feature map ever materialised |
| **Activations**, layer sequential | 8 256 B | peak of (input + output + that layer's own buffer), at `context.1.depthwise` |

There is **no single "activation memory" number** — it depends on the accelerator style, and
quoting one figure under-specifies the design.

**The layer-sequential figure is a peak taken layer by layer, not a sum of worst cases.** The
widest adjacent feature-map pair is 6 144 B (at `b1.depthwise`, whose own buffer is 40 B) and the
largest line buffer is 4 160 B (at `context.1.depthwise`, whose feature maps are small). Those
occur at *different* layers, so adding them would over-size the memory by 2 048 B. The true peak
is **8 256 B**, at `context.1.depthwise`: 2 048 in + 2 048 out + 4 160 buffer.

**There are no skip connections.** `WearSeizure1D.forward` and `DepthwiseSeparableConv1d.forward`
are purely sequential, so nothing has to stay live across a block boundary. (`MultiScaleDilatedBlock`
does concatenate two branches — but only in `multi_scale` mode; the frozen `k5_only` variant has a
single branch. A programmable design that must also run the multi-scale variant needs room for both
branch outputs at once, up to a further 4 096 B.)

Splitting a block's depthwise and pointwise into two passes **lowers** the peak rather than raising
it: at block granularity `stem → b1` needs 4 096 + 4 096 = 8 192 B, while splitting lets the stem
output be freed before b1's output is allocated, giving 6 144 B.

**Line buffer = `(kernel − 1) × dilation + 1` samples × in_channels.** Dilation, not kernel size,
drives the cost:

| Layer | k | dilation | taps | in_ch | Buffer | Share |
|---|--:|--:|--:|--:|--:|--:|
| `stem.0` | 7 | 1 | 7 | 1 | 7 B | 0.1 % |
| `b1.depthwise` | 5 | 1 | 5 | 8 | 40 B | 0.6 % |
| `b2.branch_k5` | 5 | 1 | 5 | 16 | 80 B | 1.2 % |
| `b3.branch_k5` | 5 | 2 | 9 | 24 | 216 B | 3.2 % |
| `b4.branch_k5` | 5 | 4 | 17 | 32 | 544 B | 8.0 % |
| `context.0.depthwise` | 5 | 8 | 33 | 48 | 1 584 B | 23.2 % |
| **`context.1.depthwise`** | 5 | **16** | **65** | 64 | **4 160 B** | **61.0 %** |

One layer holds 61 % of the buffer, because a k5 at dilation 16 spans 65 samples of a **32-sample**
sequence and is padded 32 each side — 64 padded samples around 32 real ones.

**Cutting it was tried and is not free.** Narrowing `context` from 64 to 16 channels saves 37 % of
the MACs and 47 % of the line buffer, and costs **2.33 pp of sensitivity**
(`docs/EXPERIMENT_LOG_G1a.md` §2h).

### How much is enough

The XC7Z020 has **140 BRAM36 blocks**, about 4 KiB of data each. What is consumed is **blocks, not
bytes** — a 544-byte buffer still occupies a whole block if it is allocated one, so fragmentation
matters more than the total. Small buffers belong in LUTRAM, not BRAM.

| Format | Total SRAM | BRAM36 blocks | % of 140 | Gate H4 (< 10 %) |
|---|--:|--:|--:|:--:|
| **INT8 / DFP8** | 18.2 KiB | ~5 | **3.6 %** | ✅ |
| **INT16 / DFP16** | 36.3 KiB | ~10 | **7.1 %** | ✅ |
| FP32, uncompressed | 72.7 KiB | ~19 | **13.6 %** | ❌ |

So: **FP32 fits the device but breaks the project's own 10 % resource gate; INT16 and INT8 both
clear it comfortably.** Since both clear it, footprint is *not* the criterion for choosing between
them — accuracy loss is, which is what `docs/PLAN_quantisation.md` measures.

The threshold that would really matter is one this design is nowhere near: **if weights did not fit
on chip**, every inference would stream them from DDR, and a DRAM access costs roughly 100× an SRAM
access and 1000× a MAC (Horowitz, ISSCC 2014). Staying on-chip is worth far more than any saving
available *within* on-chip.

And the part that is easy to get backwards here: at a **0.1 % duty cycle** the energy per hour is
dominated by **leakage of the memory that stays powered**, not by the energy of the accesses. So the
useful reading is *smaller footprint → fewer BRAM blocks powered → longer battery*, and **power-gating
the BRAMs between inferences is worth more than shrinking 36 KiB to 18 KiB**.

---

## 3. Why the published numbers are higher

The same code and the same data, 66 folds. Only the rule that partitions windows changes.

| Split | Normalised on | Threshold on | Segment sens | Accuracy | Test/train overlap |
|---|---|---|--:|--:|--:|
| **random windows** (as published) | everything | test | **0.9229** | 0.9968 | **99.6 %** |
| by recording | everything | test | 0.6173 | 0.9887 | 0.0 % |
| by recording | train only | val | **0.6033** | 0.9888 | 0.0 % |

**Splitting windows at random inflates sensitivity by 31 points.** At a 4 s window and 1 s stride
adjacent windows share 75 % of their samples, so **99.6 % of test windows have a near-duplicate in
training** — measured, not asserted. All three architectures behave the same way, so this is a
property of the protocol, not of any one model.

**And accuracy cannot see it.** Across every measured cell accuracy spans **1.15 pp** while
sensitivity spans **32 pp**. At 0.62 % ictal prevalence a model that never predicts a seizure
already scores **99.38 %** — the best cell above sits 0.30 pp higher. Accuracy has almost no
dynamic range on this data, which is why it never appears here without prevalence next to it.

Honest limits of this reproduction: it kept a 1 s stride where the published protocol slides by
**one sample**, and it evaluates segments where the published event-level figure evaluates whole
recordings. `scripts/run_stride_sweep.sh` tests the first.

---

## 4. What has been ruled out

Negative results, kept because they are the expensive kind to rediscover.

| Idea | Outcome |
|---|---|
| Wider pre-training corpus (more CHB-MIT cases) | Null at four electrode positions, **significantly worse** at one |
| Multi-channel teacher → single-channel student | **Significantly worse** false-alarm rate |
| Selecting checkpoints on AUPRC | Same sensitivity, 16 % lower FAR — an operating point, not a gain |
| Narrowing the context block (−37 % MACs) | Costs 2.33 pp sensitivity. Not free |
| Wider stages inside the MAC budget | Identical to the control: the same 72 of 77 events |

The one thing that worked is **cohort pre-training**: +6 to +9 pp, the largest effect the project
has measured. Distillation from a stronger *single-channel* teacher adds a further +1.3 pp.

Read together, the two distillation results say something reusable: **distillation helps when the
teacher's advantage is capacity, and hurts when it is information.** A soft target is only
imitable if the student can, in principle, compute it.

---

## Layout

```
configs/         Hydra groups (profile, data, split, model, window, postprocess, precision, eval)
docs/            Protocol, gates, experiment log, model card, hardware handoff, RTL interface
scripts/         CLI entry points, phase runners, diagnostics
src/wearseizure  data, signal, models, quant, postprocess, eval, training, rtl_interface
tests/           Unit (fast, synthetic) and integration smoke tests
```

| Document | What it holds |
|---|---|
| [`docs/RUNNING.md`](docs/RUNNING.md) | Install, run, servers, levers, diagnostics |
| [`docs/EXPERIMENT_LOG_G1a.md`](docs/EXPERIMENT_LOG_G1a.md) | Every real-data run in order, and what it disproved |
| [`docs/MODEL_CARD_k5only.md`](docs/MODEL_CARD_k5only.md) | Frozen model: layers, buffers, footprint, PE sizing |
| [`docs/HARDWARE_HANDOFF.md`](docs/HARDWARE_HANDOFF.md) | What the accelerator team needs, and what is still open |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | The leakage-safe protocol, in full |
| [`docs/RESEARCH_REALITY_CHECK.md`](docs/RESEARCH_REALITY_CHECK.md) | Which of this project's own earlier claims have been overturned |
