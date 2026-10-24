# Onboarding for an assisting agent

You are joining work on **WearSeizure-1D**, a single-channel EEG seizure
detector and its accelerator, aimed at IEEE TBioCAS. This file is a **map and a
set of working rules**, not a summary of results.

**It deliberately states almost no numbers.** Three briefs already exist and
each is authoritative for its own territory; a fourth document repeating their
figures would drift out of step with them, and that has already happened here
once — reconciling eleven contradictions between documents cost a day. When you
need a number, open the file named below as its source and read it there. If
you find two files disagreeing, that is a bug worth fixing, not a judgement
call to make silently.

---

## 1. The two repositories

| | |
|---|---|
| **`WearSeizure-1D`** (this one) | Ours. Model, training, evaluation, quantisation, golden model, test vectors, all the briefs. |
| **`AI-Accelerator-RTL`** (`github.com/Manhzzax/AI-Accelerator-RTL`) | A **separate team's** undergraduate thesis. They wrote the RTL. We gave them four deliverables; they imported our vectors and verified against them. Local clone at `D:\Research\AI-Accelerator-RTL`. |

**Do not push to their repository and do not refactor it.** Read it to answer
questions. The standing rule for it is: deliver exactly what is asked for,
nothing self-initiated. Their default branch is `main`; `anhquan` is also
active.

The relationship between the two is an **open question for the user to decide**,
not for us to assume: whether their design is the paper's hardware contribution
with the student as a co-author, or whether a separate implementation is
wanted. Do not plan work that presumes either answer.

---

## 2. Which document is authoritative for what

| Topic | Read | Status |
|---|---|---|
| What the paper may and may not claim | `docs/PAPER_CONTEXT_BRIEF.md` | **Binding.** §7 is the list of claims that must never appear. |
| Hardware design, gates, power, KV260 | `docs/HARDWARE_DESIGN_BRIEF.md` | **Binding.** Read §1 and §11 first. |
| Submission go/no-go | `docs/GATES.md` | Binding. |
| The frozen model, per-layer table | `docs/MODEL_CARD_k5only.md` | Authoritative for §3. §4.8 carries a struck-through DFP16 conclusion with its correction beside it — readable, but quotable by accident. |
| Every run, by row number | `docs/EXPERIMENT_LOG_G1a.md` | Authoritative history. |
| Protocol analysis, gate arithmetic | `docs/RESEARCH_REALITY_CHECK.md` | Authoritative. |
| Server paths, env vars, how to run | `docs/RUNNING.md`, `docs/SERVER_INVENTORY.md` | Current. |
| The data-and-channels argument (Vietnamese) | `docs/GIAI_TRINH_DU_LIEU.md` | Current. |

**Known stale, do not trust without checking §11 of the hardware brief:**
`src/wearseizure/rtl_interface/spec.md` (wrong bit widths),
`docs/ARCHITECTURE.md` (describes a variant the project does not deploy),
`docs/HARDWARE_HANDOFF.md`, `docs/comparison_table.pdf`.

`scripts/check_brief_consistency.py` asserts the specific ways these documents
have drifted before. **Run it after editing any brief.** It has itself been
wrong twice — both times by matching a word instead of an assertion — so if it
fails on prose that looks correct, read its check before editing the prose.

---

## 3. How work actually gets done here

**The user runs everything; you write the commands.** There is no SSH from this
machine to the GPU servers. The loop is: you write an exact command, the user
runs it on the server and pastes the output back.

Three rules that came from the user and are not negotiable:

1. **Every run command ships with its check command.** Not "run this", but "run
   this, and here is how to see whether it worked".
2. **Long jobs get a committed script, never a pasted shell blob.** The user
   has to be able to re-run and monitor it without the conversation.
3. **Enumerate before pushing.** Say what will be pushed and why, before doing
   it. This once caught synthetic weights about to be published as real.

**Communication is in Vietnamese.** Code, comments, commit messages and the
documents are in English.

### Environment traps that have cost time

* The server shell starts in conda `(base)`; the project env is `chbmit-cnn`.
* `CHBMIT_RAW_DIR` and `WEARSEIZURE_ARTIFACTS_DIR` must be set. A `.env` at the
  repo root is read by the Python entry points. **Most `scripts/*.sh` do not
  read it** — only `run_phase8_l9.sh` does. Fix the next one that bites.
* Pushing needs `gh auth switch --user NADUNGVN`; the active account is usually
  the user's work one. Switch back afterwards.
* Hex artefacts must be **LF**, forced by `write_lf()`. A stray CR in a
  `$readmemh` file looks like a hardware bug for a day.

---

## 4. The standards this project holds itself to

These are not style preferences. Most of them exist because something went
wrong once.

**Measure, do not assert.** Several confident claims here were retracted after
measurement, including two of the assistant's own recommendations. If a
statement is not backed by a number in a file, say so in the same sentence.

**Read the worst case the interval allows, not the point estimate.** Every
comparison is a patient-clustered paired bootstrap — clustered because 66 folds
come from 13 patients, and treating folds as independent narrows every interval
by pretending otherwise.

**Leakage discipline is the paper's whole contribution, so it binds us too.**
Split by EDF before filtering and normalising; causal filtering with per-EDF
state reset, never `filtfilt`; the normaliser fit on train only; thresholds
frozen on validation before test is touched. Re-fitting a threshold after
seeing test results is the exact failure the paper criticises in others. It has
been measured: it costs more than it buys.

**A metric that moves with the excerpt is not a result.** Window-level accuracy
on these vectors changes by forty points with excerpt length while the model is
identical. `scripts/demo_card.py` prints what may and may not be claimed.

**Negative results are results.** Several training levers are recorded as
measured-negative, with the mechanism. Do not quietly drop one that failed.

**Correct yourself plainly and keep going.** When you find your own error —
there are several recorded in the git history — state it in a sentence, fix it,
move on. Do not bury it and do not dwell on it.

---

## 5. Where things stand

Read these three in order; between them they are the current state.

1. `docs/PAPER_CONTEXT_BRIEF.md` §9 — what is open on the software side.
2. `docs/HARDWARE_DESIGN_BRIEF.md` §13 — the open design questions.
3. `docs/GATES.md` — what blocks submission.

The short version, as pointers rather than claims: the **software and the
verification material are closed**; the **hardware contribution is not**. The
binding gaps are measured power, bit-exactness at scale against RTL rather than
against the golden model, and a quantified contribution beyond a generic
baseline. §13 of the hardware brief lists the decisions that have to be made
before more work is useful, and several of them are the user's to make.

---

## 6. Running things locally

```bash
.venv/Scripts/python.exe -m pytest tests/unit -q        # must stay green
.venv/Scripts/python.exe scripts/check_brief_consistency.py
.venv/Scripts/python.exe scripts/verify_onset_vectors.py
.venv/Scripts/python.exe scripts/verify_onset_vectors.py --model-dir hardware/export_chb15__chb15_40
.venv/Scripts/python.exe scripts/demo_card.py
```

The verifier needs **numpy and this repository only** — no dataset, no GPU, no
PyTorch. That is deliberate: a check that runs only where the data lives cannot
be run by whoever is bringing up the hardware, so it never gets run.

Anything touching CHB-MIT runs on the server. The local `artifacts/` holds
**synthetic** results; `profile=local_synthetic` sets `enforce_gates=false`
because synthetic numbers carry no clinical meaning. Never present one as a
result.

---

## 7. Mistakes already made here, so you can skip them

Each of these is in the git history with its fix. They are listed because they
are the shapes of error this codebase invites, not because they are unusual.

* **A tolerance tighter than the file format.** A correct replay failed because
  the checker compared to 1e-9 against a column stored to eight decimals.
* **A name pattern as a discovery mechanism.** Twice. A directory of verified
  vectors sat unchecked while the scan reported green, because its name was not
  the name the glob expected. Find things by what they contain.
* **A verdict read off the wrong row.** A comparison that lost six points of
  sensitivity with an interval excluding zero printed "the data does not
  distinguish the arms", because the verdict logic only looked at latency.
* **An index from the selection, not the thing.** Selecting the second seizure
  wrote it under the first one's name and overwrote a different seizure's
  excerpt. Both files verified; only the delay revealed it.
* **A guard matching a word instead of an assertion.** Twice. The sentence
  instructing a draft *not* to use a phrase contains that phrase.
* **An error message naming a file it never read.** The runner told the user to
  configure `.env` and then checked only the environment.
* **A summary over mismatched arms.** 56 folds of one arm against 10 of
  another, printed as a difference.

The pattern underneath most of them: **something reported success while
checking less than it appeared to.** When you write a check, ask what it would
do if the thing it checks were absent.
