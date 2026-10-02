"""Guard the paper brief against contradicting itself.

Section 7 of ``docs/PAPER_CONTEXT_BRIEF.md`` is the binding list of claims that
must never reach a draft. It has already drifted out of step with the results
sections once: three seeds settled the channel ablation, section 4.4 was
rewritten, and section 7 kept the one-seed wording -- so the document
simultaneously said the equivalence claim was ruled out and that it was merely
unestablished. Nothing asserted on the replacement, so nothing noticed.

This checks the specific ways that document has gone wrong, not prose quality.
Run it after editing either brief.
"""

from __future__ import annotations

import pathlib
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
PAPER = DOCS / "PAPER_CONTEXT_BRIEF.md"
HARDWARE = DOCS / "HARDWARE_DESIGN_BRIEF.md"

# U+2212 minus and the dashes a Markdown editor substitutes all mean "minus"
# here; normalise before matching so a typographic change is not a failure.
DASHES = {"−": "-", "–": "-", "—": "-"}


def normalise(text: str) -> str:
    for src, dst in DASHES.items():
        text = text.replace(src, dst)
    return text


def section(text: str, heading: str) -> str:
    """Return one ``##`` section, heading line included."""
    start = text.index(heading)
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt == -1 else text[start:nxt]


def main() -> int:
    paper = normalise(PAPER.read_text(encoding="utf-8"))
    hardware = normalise(HARDWARE.read_text(encoding="utf-8"))

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    ablation = section(paper, "### 4.4")
    forbidden = section(paper, "## 7. Claims that must NOT")

    # The channel ablation reads the same in both places.
    for piece in ("3.41", "-5.83", "-0.93"):
        require(piece in ablation, f"section 4.4 lost the figure {piece}")
        require(piece in forbidden, f"section 7 lost the figure {piece}")
    require(
        "Measured false" in forbidden,
        "section 7 no longer calls the one-channel equivalence claim measured false",
    )
    require(
        "absence of evidence" in forbidden,
        "section 7 lost the note that the one-seed interval was not evidence of equivalence",
    )

    # One-seed figures that three seeds replaced must not come back.
    for stale in ("10.53", "19.7 points of segment", "8.5 points of AUROC"):
        require(stale not in paper, f"a one-seed figure reappeared: {stale}")

    # Hardware is in scope and unbuilt; section 7 has to say so, because a
    # drafting agent given a design section will otherwise report results for it.
    require(
        "Any hardware result whatsoever" in forbidden,
        "section 7 no longer forbids unmeasured hardware results",
    )
    require(
        "claim 3" in paper,
        "section 1 no longer states the hardware contribution as a claim",
    )

    # The power path is whole-SOM only; neither brief may promise a PL-only figure.
    require(
        "VCC_SOM" in hardware,
        "the hardware brief lost the KV260 power-rail finding",
    )
    require(
        "differential" in hardware.lower(),
        "the hardware brief lost the differential power-measurement protocol",
    )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("briefs agree: ablation, hardware scope, and the power path are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
