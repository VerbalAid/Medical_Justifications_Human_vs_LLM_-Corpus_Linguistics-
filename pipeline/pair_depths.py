#!/usr/bin/env python3
# Same *surface* entity string in corpus A vs corpus B, then SNOMED depth + histogram.
#
# The NER CSV is built from **pooled** case files per side (one row per entity string per corpus),
# so this pairing is **not** restricted to ``the same token in the same case file''. For that
# mention-aligned slice (strict twins + per-case string intersection), run
# ``python -m pipeline.per_case_same_string_ner`` after ``ner_depth``.

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import SNOMED_DEPTH_NER, SNOMED_PAIR_DEPTH_CSV, SNOMED_PAIR_HIST, ensure_results, explain_missing_ner_csv

DEFAULT_NER_CSV = SNOMED_DEPTH_NER
OUT_CSV = SNOMED_PAIR_DEPTH_CSV
OUT_HIST = SNOMED_PAIR_HIST


def parse_depth(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s or not s.isdigit():
        return None
    return int(s)


def load_rows_by_entity(ner_csv: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_a: dict[str, dict] = {}
    by_b: dict[str, dict] = {}
    with ner_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ent = (row.get("entity") or "").strip()
            if not ent:
                continue
            corp = row.get("corpus", "")
            if corp == "A":
                by_a[ent] = row
            elif corp == "B":
                by_b[ent] = row
    return by_a, by_b


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(description="Depth pairing A vs B from NER CSV.")
    parser.add_argument("--ner-csv", type=Path, default=DEFAULT_NER_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-hist", type=Path, default=OUT_HIST)
    args = parser.parse_args()

    if not args.ner_csv.is_file():
        raise SystemExit(explain_missing_ner_csv(args.ner_csv))

    by_a, by_b = load_rows_by_entity(args.ner_csv)
    common = sorted(set(by_a) & set(by_b))

    rows_out: list[dict[str, str | int]] = []
    diffs: list[int] = []

    for ent in common:
        ra, rb = by_a[ent], by_b[ent]
        ca = (ra.get("snomed_concept") or "").strip()
        cb = (rb.get("snomed_concept") or "").strip()
        if not ca or not cb:
            continue
        da = parse_depth(ra.get("depth", ""))
        db = parse_depth(rb.get("depth", ""))
        if da is None or db is None:
            continue
        diff = db - da
        diffs.append(diff)
        snomed_one = ca if ca == cb else f"{ca};{cb}"
        rows_out.append(
            {
                "entity": ent,
                "snomed_concept": snomed_one,
                "depth_a": da,
                "depth_b": db,
                "depth_diff": diff,
            }
        )

    fieldnames = ["entity", "snomed_concept", "depth_a", "depth_b", "depth_diff"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    n = len(rows_out)
    if n == 0:
        print("No paired entities with concept and numeric depth in both corpora.")
        print(f"Wrote empty {args.out_csv.name}")
        return

    mean_diff = sum(diffs) / n
    n_b_shallow = sum(1 for d in diffs if d < 0)
    n_b_deep = sum(1 for d in diffs if d > 0)
    n_eq = sum(1 for d in diffs if d == 0)

    print(f"Paired entities: {n}")
    print(f"Mean depth difference (B − A): {mean_diff:.4f}")
    print(f"B shallower than A (diff < 0): {n_b_shallow}")
    print(f"B deeper than A (diff > 0): {n_b_deep}")
    print(f"Equal depth: {n_eq}")
    print(f"Wrote {args.out_csv}")

    if n_eq == n:
        print(
            "Note: all depth differences are 0. The same surface string in A and B usually maps to "
            "the same SNOMED concept, so depth matches by definition. For pairwise comparison across "
            "related concepts on shared branches, use: python -m pipeline.branch_pairs",
            file=sys.stderr,
        )

    plt.figure(figsize=(8, 5))
    plt.hist(diffs, bins=min(40, max(10, n // 5)), color="#455A64", edgecolor="white", alpha=0.9)
    plt.xlabel("Depth difference (corpus B − corpus A)")
    plt.ylabel("Count")
    plt.title("SNOMED depth difference for shared entity strings")
    plt.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.8)
    plt.tight_layout()
    plt.savefig(args.out_hist, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out_hist}")


if __name__ == "__main__":
    main()
