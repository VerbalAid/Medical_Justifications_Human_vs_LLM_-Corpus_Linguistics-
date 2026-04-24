#!/usr/bin/env python3
"""Word-count summary for corpus_a / corpus_b (per file and corpus-wide)."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from .paths import CORPUS_A, CORPUS_B


def word_count(text: str) -> int:
    return len(text.split())


def counts_in_folder(folder: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for p in sorted(folder.glob("case_*.txt")):
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        out.append((p.name, word_count(t)))
    return out


def summarise(label: str, rows: list[tuple[str, int]]) -> None:
    if not rows:
        print(f"{label}: no case_*.txt files.")
        return
    vals = [n for _, n in rows]
    print(f"\n{label} ({len(rows)} files)")
    print(f"  words per file — mean: {statistics.mean(vals):.1f}  median: {statistics.median(vals):.0f}")
    print(f"  min: {min(vals)}  max: {max(vals)}")
    print(f"  total words: {sum(vals):,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Word counts for corpus_a and corpus_b case files.")
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A, help="Override path to human folder.")
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B, help="Override path to model folder.")
    p.add_argument(
        "--per-file",
        action="store_true",
        help="Print every case_*.txt word count (can be long).",
    )
    args = p.parse_args()

    a_rows = counts_in_folder(args.corpus_a)
    b_rows = counts_in_folder(args.corpus_b)
    summarise("Corpus A (human)", a_rows)
    summarise("Corpus B (model)", b_rows)
    if a_rows and b_rows and len(a_rows) == len(b_rows):
        deltas = [b - a for (_, a), (_, b) in zip(a_rows, b_rows)]
        print("\nB minus A (words per file, same order)")
        print(f"  mean delta: {statistics.mean(deltas):+.1f}  median: {statistics.median(deltas):+.0f}")
    if args.per_file:
        print("\nPer-file (A)")
        for name, n in a_rows:
            print(f"  {name}\t{n}")
        print("\nPer-file (B)")
        for name, n in b_rows:
            print(f"  {name}\t{n}")


if __name__ == "__main__":
    main()
