#!/usr/bin/env python3
"""Stratified 40-case list for unkeyed Corpus B ablation (matches Table 2 areas)."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from .paths import CORPUS_A, CORPUS_B, HEDGE_CASE_TABLE, ROOT

QUOTAS: dict[str, int] = {
    "General medicine (no clear lexical match)": 18,
    "Obstetrics & gynaecology": 6,
    "Nephrology & urology": 6,
    "Respiratory medicine": 6,
    "Cardiovascular medicine": 4,
}


def _ok_pair(name: str) -> bool:
    pa, pb = CORPUS_A / name, CORPUS_B / name
    if not pa.is_file() or not pb.is_file():
        return False
    return bool(pa.read_text(encoding="utf-8").strip()) and bool(pb.read_text(encoding="utf-8").strip())


def pick_cases(*, seed: int = 42) -> list[str]:
    by_area: dict[str, list[str]] = defaultdict(list)
    if not HEDGE_CASE_TABLE.is_file():
        raise SystemExit(f"Missing {HEDGE_CASE_TABLE}\nRun: python -m pipeline.hedging_area")
    with HEDGE_CASE_TABLE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("case_file") or "").strip()
            area = (row.get("medical_area") or "").strip()
            if area not in QUOTAS or not name:
                continue
            if _ok_pair(name):
                by_area[area].append(name)

    rng = random.Random(seed)
    picked: list[str] = []
    for area, quota in QUOTAS.items():
        pool = list(by_area[area])
        rng.shuffle(pool)
        picked.extend(pool[: min(quota, len(pool))])

    need = 40 - len(picked)
    if need > 0:
        gen = [x for x in by_area["General medicine (no clear lexical match)"] if x not in picked]
        rng.shuffle(gen)
        for x in gen:
            if need <= 0:
                break
            picked.append(x)
            need -= 1

    picked = sorted(picked[:40])
    if len(picked) < 40:
        raise SystemExit(f"Only {len(picked)} cases available for stratified sample (need 40).")
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description="Write stratified 40-case list for unkeyed ablation.")
    p.add_argument("--out", type=Path, default=ROOT / "cases_unkeyed40.txt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    picked = pick_cases(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"Wrote {len(picked)} lines → {args.out}", file=sys.stderr)
    area_by_file: dict[str, str] = {}
    with HEDGE_CASE_TABLE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            area_by_file[(row.get("case_file") or "").strip()] = (row.get("medical_area") or "").strip()
    counts: dict[str, int] = defaultdict(int)
    for x in picked:
        counts[area_by_file.get(x, "?")] += 1
    print("Area counts:", dict(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
