#!/usr/bin/env python3
# Count a fixed list of hedge-like words; output rate per 1000 tokens for corpus A and B.

from __future__ import annotations

import re
from pathlib import Path

from .paths import CORPUS_A, CORPUS_B, HEDGE_SUMMARY, ROOT, ensure_results

HEDGING = frozenset(
    """
    may might could would can
    often usually typically generally sometimes frequently rarely
    possibly probably perhaps seemingly apparently
    suggest suggests suggesting suggested
    seem seems seemed appear appears appeared
    likely unlikely potential potentially
    generally broadly roughly approximately
    """.split()
)

TOKEN_RE = re.compile(r"\b[a-z]+\b", re.IGNORECASE)


def hedge_hits_and_tokens(text: str) -> tuple[int, int]:
    hedge_hits = total = 0
    for m in TOKEN_RE.finditer(text):
        total += 1
        if m.group(0).lower() in HEDGING:
            hedge_hits += 1
    return hedge_hits, total


def read_corpus_tokens(folder: Path):
    hedge_hits = total = 0
    for p in sorted(folder.glob("case_*.txt")):
        h, t = hedge_hits_and_tokens(p.read_text(encoding="utf-8"))
        hedge_hits += h
        total += t
    return hedge_hits, total


def main() -> None:
    ensure_results()
    ha, ta = read_corpus_tokens(CORPUS_A)
    hb, tb = read_corpus_tokens(CORPUS_B)
    ra = (1000.0 * ha / ta) if ta else 0.0
    rb = (1000.0 * hb / tb) if tb else 0.0

    HEDGE_SUMMARY.write_text(
        "\n".join(
            [
                "corpus,hedge_hits,total_tokens,per_1000_tokens",
                f"A_human,{ha},{ta},{ra:.4f}",
                f"B_model,{hb},{tb},{rb:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Corpus A: {ha} hedging / {ta} tokens → {ra:.2f} per 1,000")
    print(f"Corpus B: {hb} hedging / {tb} tokens → {rb:.2f} per 1,000")
    print(f"wrote {HEDGE_SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
