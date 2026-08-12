# KPI gates

Source: Research Decision Memo, Table 6. Encoded in `configs/eval/gates.yaml`
and checked by `eval/report.check_gates` / `scripts/evaluate.py`.

Three levels per KPI: **minimum** (gate to continue), **target** (must hit
before writing the TBioCAS paper), **stretch** (pursue only if it doesn't
delay hardware validation).

| KPI | Minimum | Target | Stretch |
|---|---|---|---|
| Personalized event sensitivity | >=97.0% | >=98.5% | >=99.0% |
| FAR | <=0.30/h | <=0.20/h | <=0.10/h |
| Detection delay | mean <=5s | mean <=4s, median <=3s | mean <=3s, median <=2s |
| Worst patient | sens >=85%, FAR <=1.0/h | sens >=90%, FAR <1.0/h | sens >=95%, FAR <=0.5/h |
| Zero-shot LOSO | sens >=85%, FAR <=0.75/h | sens >=90%, FAR <=0.50/h | sens >=92%, FAR <=0.30/h |
| Model budget | <=32k params, <=2M MAC | <=16k, <=0.70M | <=12k, <=0.50M |
| INT8 loss vs FP32 | <=1.0pp | <=0.5pp | W4A8 loss <=1.0pp |
| Continuous test exposure | >=100h | >=150h | >=200h + external corpus |

Hardware KPIs (RTL latency/throughput/Fmax/resources/power/energy) are in the
same table in the memo but are out of scope until Gate G3+ (real board).

## Enforcement

`profile.enforce_gates` (see `configs/profile/*.yaml`) is `false` under
`local_synthetic` -- synthetic data has no clinical meaning, so a gate
"failure" there is expected and non-fatal, only informative about pipeline
wiring. It is `true` under `server`: `scripts/evaluate.py` raises if any
gated metric is `below_minimum` once real data is in use.

## Hard go/no-go (memo, "TIEU CHI GO/NO-GO CUNG")

Do not submit without: leakage-safe split, continuous event test, INT8-RTL
bit-exact verification, measured FPGA power, same-protocol comparison, and at
least one quantified hardware contribution beyond a generic HLS/CPU baseline.
None of the hardware items are satisfiable until Gate G3+ (RTL/FPGA), which
is intentionally out of scope for this repository until server/board details
are confirmed.
