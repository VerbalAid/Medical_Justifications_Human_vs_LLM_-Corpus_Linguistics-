#!/usr/bin/env python3
"""
Build many (entity_a, entity_b) rows from NER on both corpora.

This is a full cross-product: every SNOMED-linked span from human files
against every SNOMED-linked span from model files. If the two concepts share
any ancestor, we pick the *deepest* common one as the LCA (within a hop cap).

So two strings are compared because they appeared *somewhere* in the two sides
and sit under the same broad branch — not because the pipeline matched them as
the "same" word. For same-surface-string depth compare, use ``pair_depths``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict, deque
from pathlib import Path

from .paths import (
    SNOMED_BRANCH_CATEGORIES,
    SNOMED_BRANCH_PAIRS,
    SNOMED_DEPTH_NER,
    ensure_results,
    explain_missing_ner_csv,
)
from .rf2 import (
    DEFAULT_RF2_ROOT,
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)

DEFAULT_NER_CSV = SNOMED_DEPTH_NER
OUT_CSV = SNOMED_BRANCH_PAIRS
OUT_CATEGORY_CSV = SNOMED_BRANCH_CATEGORIES
MAX_HOPS_TO_LCA = 5

# root metadata children -> label for "which hierarchy" (closest ancestor wins if tie)
SNOMED_TOP_LEVEL: tuple[tuple[str, str], ...] = (
    ("404684003", "Clinical finding"),
    ("71388002", "Procedure"),
    ("123037004", "Body structure"),
    ("410607006", "Organism"),
    ("105590001", "Substance"),
    ("363787002", "Observable entity"),
    ("272379006", "Event"),
    ("243796009", "Situation with explicit context"),
    ("48176007", "Social context"),
    ("308916002", "Environment or geographical location"),
    ("786210002", "Physical object"),
    ("260787004", "Physical force"),
    ("49755003", "Morphologically abnormal structure"),
    ("123038009", "Specimen"),
    ("734139008", "Treatment"),
    ("370115009", "Special concept"),
    ("419891003", "Record artifact"),
    ("373873005", "Pharmaceutical / biologic product"),
)
TOP_LEVEL_IDS: frozenset[str] = frozenset(tid for tid, _ in SNOMED_TOP_LEVEL)
TOP_LEVEL_LABEL: dict[str, str] = {tid: name for tid, name in SNOMED_TOP_LEVEL}


def hierarchy_category_for_concept(
    concept_id: str,
    anc: frozenset[str],
    parents: dict[str, list[str]],
    hop_cap: int = 250,
) -> str:
    """
    Map a concept to a top-level SNOMED hierarchy name using ancestor closure.
    """
    if concept_id == RF2_ROOT_CONCEPT_ID:
        return "SNOMED root"
    candidates = [tid for tid in TOP_LEVEL_IDS if tid in anc]
    if not candidates:
        return "Other or unclassified"
    if len(candidates) == 1:
        return TOP_LEVEL_LABEL[candidates[0]]

    best_tid = candidates[0]
    best_hops: int | None = None
    for tid in candidates:
        h = min_hops_upward(concept_id, tid, parents, hop_cap)
        if h is None:
            continue
        if best_hops is None or h < best_hops:
            best_hops = h
            best_tid = tid
    if best_hops is None:
        return TOP_LEVEL_LABEL[candidates[0]]
    return TOP_LEVEL_LABEL[best_tid]


def ensure_anc_in_cache(
    concept_ids: set[str],
    anc_cache: dict[str, frozenset[str]],
    parents: dict[str, list[str]],
) -> None:
    for cid in concept_ids:
        if cid not in anc_cache:
            anc_cache[cid] = ancestor_closure(cid, parents)


def aggregate_category_breakdown(
    pairs: list[dict[str, str | int]],
    anc_cache: dict[str, frozenset[str]],
    parents: dict[str, list[str]],
) -> list[dict[str, float | int | str]]:
    """Per LCA semantic category: counts and % where A deeper, B deeper, equal depth."""
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "a_deeper": 0, "b_deeper": 0, "equal": 0}
    )
    for r in pairs:
        lca = str(r["lca_concept"])
        anc = anc_cache[lca]
        cat = hierarchy_category_for_concept(lca, anc, parents)
        da, db = int(r["depth_a"]), int(r["depth_b"])
        s = stats[cat]
        s["total"] += 1
        if da > db:
            s["a_deeper"] += 1
        elif db > da:
            s["b_deeper"] += 1
        else:
            s["equal"] += 1

    rows: list[dict[str, float | int | str]] = []
    for cat in sorted(stats.keys(), key=lambda c: (-stats[c]["total"], c)):
        s = stats[cat]
        t = s["total"]
        rows.append(
            {
                "category": cat,
                "total_pairs": t,
                "n_a_deeper": s["a_deeper"],
                "n_b_deeper": s["b_deeper"],
                "n_equal_depth": s["equal"],
                "pct_a_deeper": round(100.0 * s["a_deeper"] / t, 2) if t else 0.0,
                "pct_b_deeper": round(100.0 * s["b_deeper"] / t, 2) if t else 0.0,
                "pct_equal_depth": round(100.0 * s["equal"] / t, 2) if t else 0.0,
            }
        )
    return rows


def print_category_table(rows: list[dict[str, float | int | str]]) -> None:
    if rows:
        mlen = max(len(str(r["category"])) for r in rows)
    else:
        mlen = 10
    w_cat = max(len("category"), mlen)
    hdr = (
        f"{'category':<{w_cat}}  {'pairs':>8}  {'%A deeper':>10}  {'%B deeper':>10}  {'%equal':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['category']:<{w_cat}}  {r['total_pairs']:>8}  "
            f"{r['pct_a_deeper']:>9.2f}%  {r['pct_b_deeper']:>9.2f}%  "
            f"{r['pct_equal_depth']:>7.2f}%"
        )


def parse_depth(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s or not s.isdigit():
        return None
    return int(s)


def load_corpus_rows(ner_csv: Path) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """Lists of (entity, concept_id, depth) with concept and depth present."""
    a: list[tuple[str, str, int]] = []
    b: list[tuple[str, str, int]] = []
    with ner_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("snomed_concept") or "").strip()
            d = parse_depth(row.get("depth", ""))
            ent = (row.get("entity") or "").strip()
            if not cid or d is None or not ent:
                continue
            corp = row.get("corpus", "")
            if corp == "A":
                a.append((ent, cid, d))
            elif corp == "B":
                b.append((ent, cid, d))
    return a, b


def ancestor_closure(
    concept: str, parents: dict[str, list[str]]
) -> frozenset[str]:
    """All concepts reachable upward from concept (including itself)."""
    out: set[str] = set()
    stack = [concept]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        for p in parents.get(n, []):
            stack.append(p)
    return frozenset(out)


def min_hops_upward(
    start: str,
    target: str,
    parents: dict[str, list[str]],
    hop_limit: int,
) -> int | None:
    """Minimum parent-walk hops from start to target (target must be an ancestor)."""
    if start == target:
        return 0
    q: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while q:
        n, h = q.popleft()
        if h >= hop_limit:
            continue
        for p in parents.get(n, []):
            if p == target:
                return h + 1
            if p not in seen:
                seen.add(p)
                q.append((p, h + 1))
    return None


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(
        description="Cross every NER entity in A with every NER entity in B; keep pairs that share an LCA (hop-limited)."
    )
    parser.add_argument(
        "--rf2-root",
        type=Path,
        default=DEFAULT_RF2_ROOT,
        help="SNOMED RF2 root (default: project layout in pipeline.paths).",
    )
    parser.add_argument(
        "--ner-csv",
        type=Path,
        default=DEFAULT_NER_CSV,
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=MAX_HOPS_TO_LCA,
        help="Max is-a hops from each concept up to LCA (default 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_CSV,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200_000,
        help="Print progress to stderr every N inner iterations (default 200000).",
    )
    parser.add_argument(
        "--dedupe-concept-pairs",
        action="store_true",
        help="Keep one row per (concept_a, concept_b), keeping the entity pair with largest |depth_diff|.",
    )
    parser.add_argument(
        "--category-out",
        type=Path,
        default=OUT_CATEGORY_CSV,
        help="Semantic category breakdown CSV (default: snomed_category_breakdown.csv).",
    )
    args = parser.parse_args()

    if not args.ner_csv.is_file():
        raise SystemExit(explain_missing_ner_csv(args.ner_csv))

    rows_a, rows_b = load_corpus_rows(args.ner_csv)
    if not rows_a or not rows_b:
        raise SystemExit("Need non-empty matched rows with depth in both corpora.")

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))
    if not rf2_root.is_dir():
        raise SystemExit(f"RF2 root must be a directory: {rf2_root}")

    _, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not rel_files:
        raise SystemExit("No sct2_Relationship *.txt under --rf2-root.")

    print("Loading RF2 is-a …", file=sys.stderr)
    child_to_parents = load_rf2_isa_parent_map(rel_files)
    depth_memo: dict[str, int | None] = {}

    def depth_of(cid: str) -> int | None:
        return max_depth_from_root_rf2(cid, child_to_parents, RF2_ROOT_CONCEPT_ID, depth_memo)

    print("Caching ancestor sets for concepts in CSV …", file=sys.stderr)
    concepts: set[str] = set()
    for _, c, _ in rows_a:
        concepts.add(c)
    for _, c, _ in rows_b:
        concepts.add(c)
    anc_cache: dict[str, frozenset[str]] = {c: ancestor_closure(c, child_to_parents) for c in concepts}

    pairs_out: list[dict[str, str | int]] = []
    dedupe_best: dict[tuple[str, str], dict[str, str | int]] = {}
    inner = 0
    na = len(rows_a)
    nb = len(rows_b)

    def record_pair(rec: dict[str, str | int]) -> None:
        if args.dedupe_concept_pairs:
            k = (str(rec["concept_a"]), str(rec["concept_b"]))
            prev = dedupe_best.get(k)
            ad = abs(int(rec["depth_b"]) - int(rec["depth_a"]))
            if prev is None or ad > abs(int(prev["depth_b"]) - int(prev["depth_a"])):
                dedupe_best[k] = rec
        else:
            pairs_out.append(rec)

    for ia, (ea, ca, da) in enumerate(rows_a):
        if ia % 50 == 0:
            print(f"  Outer {ia}/{na} …", file=sys.stderr)
        anc_a = anc_cache[ca]
        for eb, cb, db in rows_b:
            inner += 1
            if args.progress_every and inner % args.progress_every == 0:
                print(f"  … {inner} pair checks, {len(pairs_out)} kept", file=sys.stderr)

            anc_b = anc_cache[cb]
            if len(anc_a) <= len(anc_b):
                common = [x for x in anc_a if x in anc_b]
            else:
                common = [x for x in anc_b if x in anc_a]
            if not common:
                continue

            best_lca: str | None = None
            best_d = -1
            for x in common:
                dx = depth_of(x)
                if dx is not None and dx > best_d:
                    best_d = dx
                    best_lca = x
            if best_lca is None:
                continue

            ha = min_hops_upward(ca, best_lca, child_to_parents, args.max_hops)
            hb = min_hops_upward(cb, best_lca, child_to_parents, args.max_hops)
            if ha is None or hb is None:
                continue
            if ha > args.max_hops or hb > args.max_hops:
                continue

            depth_lca = depth_of(best_lca)
            if depth_lca is None:
                continue

            record_pair(
                {
                    "entity_a": ea,
                    "entity_b": eb,
                    "concept_a": ca,
                    "concept_b": cb,
                    "depth_a": da,
                    "depth_b": db,
                    "lca_concept": best_lca,
                    "depth_lca": depth_lca,
                    "hops_a_to_lca": ha,
                    "hops_b_to_lca": hb,
                    "depth_diff_b_minus_a": db - da,
                }
            )

    if args.dedupe_concept_pairs:
        pairs_out = list(dedupe_best.values())

    fieldnames = [
        "entity_a",
        "entity_b",
        "concept_a",
        "concept_b",
        "depth_a",
        "depth_b",
        "lca_concept",
        "depth_lca",
        "hops_a_to_lca",
        "hops_b_to_lca",
        "depth_diff_b_minus_a",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in pairs_out:
            w.writerow(r)

    n_pairs = len(pairs_out)
    print(f"Branch pairs found (LCA within {args.max_hops} hops each side): {n_pairs}")
    if n_pairs == 0:
        print(f"Wrote {args.out}")
        return

    n_a_deeper = sum(1 for r in pairs_out if r["depth_a"] > r["depth_b"])
    pct = 100.0 * n_a_deeper / n_pairs
    print(f"Pairs where Corpus A is deeper (more specific) than B: {n_a_deeper} ({pct:.2f}%)")
    print(f"Wrote {args.out}")

    lca_ids = {str(r["lca_concept"]) for r in pairs_out}
    print(f"Caching ancestor sets for {len(lca_ids):,} distinct LCAs (category labels) …", file=sys.stderr)
    ensure_anc_in_cache(lca_ids, anc_cache, child_to_parents)

    breakdown_rows = aggregate_category_breakdown(pairs_out, anc_cache, child_to_parents)
    all_a = sum(1 for r in pairs_out if r["depth_a"] > r["depth_b"])
    all_b = sum(1 for r in pairs_out if r["depth_b"] > r["depth_a"])
    all_eq = n_pairs - all_a - all_b
    breakdown_rows.append(
        {
            "category": "ALL",
            "total_pairs": n_pairs,
            "n_a_deeper": all_a,
            "n_b_deeper": all_b,
            "n_equal_depth": all_eq,
            "pct_a_deeper": round(100.0 * all_a / n_pairs, 2),
            "pct_b_deeper": round(100.0 * all_b / n_pairs, 2),
            "pct_equal_depth": round(100.0 * all_eq / n_pairs, 2),
        }
    )

    print("\nSemantic category breakdown (by LCA top-level hierarchy):")
    print_category_table(breakdown_rows)

    bfieldnames = [
        "category",
        "total_pairs",
        "n_a_deeper",
        "n_b_deeper",
        "n_equal_depth",
        "pct_a_deeper",
        "pct_b_deeper",
        "pct_equal_depth",
    ]
    with args.category_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bfieldnames)
        w.writeheader()
        for row in breakdown_rows:
            w.writerow(row)
    print(f"Wrote {args.category_out}")


if __name__ == "__main__":
    main()
