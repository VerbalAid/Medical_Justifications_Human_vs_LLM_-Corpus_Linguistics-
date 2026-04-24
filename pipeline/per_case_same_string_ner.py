#!/usr/bin/env python3
"""Per strict case pair: NER on A and B, keep entity strings that appear in *both* files, map to SNOMED depth.

This is the mention-aligned ``same token in both explanations'' slice the paper contrasts with
pooled ``pair_depths'' (corpus-wide string intersection). One surface string still maps to one
RF2 concept in our pipeline, so ``depth_a'' and ``depth_b'' are usually identical here; the CSV is
meant as a **candidate list** for human or LLM review (same referent / specificity ladder) before
trusting any depth story.
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
from .paths import CORPUS_A, CORPUS_B, DEFAULT_RF2_ROOT, SNOMED_PER_CASE_SAME_STRING, ensure_results
from .rf2 import (
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "For each case_*.txt twin in corpus_a / corpus_b: NER both texts, intersect entity strings, "
            "attach SNOMED depth (same string → same concept in this matcher). Writes a long CSV for "
            "audit or downstream LLM gating."
        )
    )
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--out-csv", type=Path, default=SNOMED_PER_CASE_SAME_STRING)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
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
    for path_a in sorted(args.corpus_a.glob("case_*.txt")):
        path_b = args.corpus_b / path_a.name
        if not path_b.is_file():
            continue
        try:
            text_a = path_a.read_text(encoding="utf-8", errors="replace")
            text_b = path_b.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ca = collect_ner_entities_for_document(text_a, nlp)
        cb = collect_ner_entities_for_document(text_b, nlp)
        common = sorted(set(ca) & set(cb))
        stem = path_a.name
        for ent in common:
            cid = best_snomed_match(ent, exact_map, rows, inv)
            if not cid:
                continue
            d = max_depth_from_root_rf2(cid, parents, RF2_ROOT_CONCEPT_ID, depth_memo)
            if d is None:
                continue
            excerpt_a = context_from_case_file(args.corpus_a, stem, ent)
            excerpt_b = context_from_case_file(args.corpus_b, stem, ent)
            out_rows.append(
                {
                    "case_file": stem,
                    "entity": ent,
                    "snomed_concept": cid,
                    "depth": int(d),
                    "count_in_case_a": int(ca[ent]),
                    "count_in_case_b": int(cb[ent]),
                    "context_sentence_a": excerpt_a,
                    "context_sentence_b": excerpt_b,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_file",
        "entity",
        "snomed_concept",
        "depth",
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
