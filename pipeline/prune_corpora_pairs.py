#!/usr/bin/env python3
# Delete case_*.txt if the twin file is missing or only whitespace on the other side.

from __future__ import annotations
import argparse
from pathlib import Path

from .paths import CORPUS_A, CORPUS_B, ROOT


def _is_nonempty_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return bool(path.read_text(encoding="utf-8").strip())


def prune_paired_corpora() -> tuple[int, int]:
    stems_a = {p.name for p in CORPUS_A.glob("case_*.txt")}
    stems_b = {p.name for p in CORPUS_B.glob("case_*.txt")}
    all_stems = stems_a | stems_b
    pairs_kept = 0
    stems_pruned = 0
    for name in sorted(all_stems):
        pa = CORPUS_A / name
        pb = CORPUS_B / name
        if _is_nonempty_file(pa) and _is_nonempty_file(pb):
            pairs_kept += 1
            continue
        stems_pruned += 1
        if pa.is_file():
            pa.unlink()
        if pb.is_file():
            pb.unlink()
    return pairs_kept, stems_pruned


def main() -> None:
    k, r = prune_paired_corpora()
    print(f"Strict pairs: {k} kept under {CORPUS_A.relative_to(ROOT)} / {CORPUS_B.relative_to(ROOT)}.")
    if r:
        print(f"Pruned {r} stem(s) (missing side or empty file).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Keep only non-empty A/B case file pairs.")
    p.parse_args()
    main()
