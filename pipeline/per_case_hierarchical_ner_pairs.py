#!/usr/bin/env python3
"""Per case file: NER on human (A) and model (B) texts, emit pairs of *different* strings whose SNOMED concepts stand in a proper is-a relationship (broader vs more specific).

Unlike ``per_case_same_string_ner`` (intersection of identical surface strings), this script builds
cross-side entity pairs where one concept is a strict ancestor of the other in RF2, and the depth
gap meets a minimum so the specificity contrast is visible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .corpus_context import context_from_case_file
from .ner_depth import (
    _ensure_python,
    _load_nlp,
    best_snomed_match,
    collect_ner_entities_for_document,
    index_description_rows,
    load_rf2_en_description_rows,
    load_rf2_term_to_concept,
)
from .paths import CORPUS_A, CORPUS_B, DEFAULT_RF2_ROOT, SNOMED_PER_CASE_HIERARCHICAL, ensure_results
from .rf2 import (
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    is_proper_isa_ancestor,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Case-aligned human vs model NER pairs: different strings, SNOMED is-a ancestor/descendant, "
            "with a minimum depth gap for a clear broad-vs-specific ladder."
        )
    )
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--out-csv", type=Path, default=SNOMED_PER_CASE_HIERARCHICAL)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument(
        "--min-depth-gap",
        type=int,
        default=3,
        metavar="D",
        help="Minimum |depth_a - depth_b| (RF2 longest path to root) to keep a pair.",
    )
    p.add_argument(
        "--max-pairs-per-case",
        type=int,
        default=20,
        metavar="N",
        help="After sorting by depth gap, keep at most N pairs per case file.",
    )
    args = p.parse_args()

    ensure_results()
    _ensure_python()

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    inv = index_description_rows(rows)
    parents = load_rf2_isa_parent_map(rel_files)
    depth_memo: dict[str, int | None] = {}

    print("Loading en_core_sci_lg …", file=sys.stderr)
    nlp = _load_nlp()

    out_rows: list[dict[str, object]] = []
    min_gap = max(1, args.min_depth_gap)
    cap = max(1, args.max_pairs_per_case)

    for path_a in sorted(args.corpus_a.glob("case_*.txt")):
        path_b = args.corpus_b / path_a.name
        if not path_b.is_file():
            continue
        try:
            text_a = path_a.read_text(encoding="utf-8", errors="replace")
            text_b = path_b.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ca_counts = collect_ner_entities_for_document(text_a, nlp)
        cb_counts = collect_ner_entities_for_document(text_b, nlp)
        stem = path_a.name

        resolved_a: list[tuple[str, str, int]] = []
        for ent in sorted(set(ca_counts)):
            cid = best_snomed_match(ent, exact_map, rows, inv)
            if not cid:
                continue
            d = max_depth_from_root_rf2(cid, parents, RF2_ROOT_CONCEPT_ID, depth_memo)
            if d is None:
                continue
            resolved_a.append((ent, cid, int(d)))

        resolved_b: list[tuple[str, str, int]] = []
        for ent in sorted(set(cb_counts)):
            cid = best_snomed_match(ent, exact_map, rows, inv)
            if not cid:
                continue
            d = max_depth_from_root_rf2(cid, parents, RF2_ROOT_CONCEPT_ID, depth_memo)
            if d is None:
                continue
            resolved_b.append((ent, cid, int(d)))

        candidates: list[tuple[int, dict[str, object]]] = []
        for ea, c_a, d_a in resolved_a:
            for eb, c_b, d_b in resolved_b:
                if ea.strip().lower() == eb.strip().lower():
                    continue
                gap = abs(d_a - d_b)
                if gap < min_gap:
                    continue
                human_is_broader = is_proper_isa_ancestor(parents, c_a, c_b)
                model_is_broader = is_proper_isa_ancestor(parents, c_b, c_a)
                if not human_is_broader and not model_is_broader:
                    continue
                rel = "human_is_broader" if human_is_broader else "model_is_broader"
                excerpt_a = context_from_case_file(args.corpus_a, stem, ea)
                excerpt_b = context_from_case_file(args.corpus_b, stem, eb)
                row = {
                    "case_file": stem,
                    "entity_a": ea,
                    "entity_b": eb,
                    "snomed_concept_a": c_a,
                    "snomed_concept_b": c_b,
                    "depth_a": d_a,
                    "depth_b": d_b,
                    "depth_gap_abs": gap,
                    "isa_relation": rel,
                    "count_in_case_a": int(ca_counts[ea]),
                    "count_in_case_b": int(cb_counts[eb]),
                    "context_sentence_a": excerpt_a,
                    "context_sentence_b": excerpt_b,
                }
                candidates.append((gap, row))

        candidates.sort(key=lambda t: (-t[0], (t[1]["entity_a"] or "")[:40], (t[1]["entity_b"] or "")[:40]))
        for _, row in candidates[:cap]:
            out_rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_file",
        "entity_a",
        "entity_b",
        "snomed_concept_a",
        "snomed_concept_b",
        "depth_a",
        "depth_b",
        "depth_gap_abs",
        "isa_relation",
        "count_in_case_a",
        "count_in_case_b",
        "context_sentence_a",
        "context_sentence_b",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {len(out_rows)} row(s) to {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
