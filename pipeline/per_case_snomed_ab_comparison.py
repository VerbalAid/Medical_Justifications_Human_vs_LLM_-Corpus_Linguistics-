#!/usr/bin/env python3
"""Per case file: map *all* NER entities from corpus A and corpus B to SNOMED, then compare sets.

Unlike ``per_case_same_string_ner`` (intersection of identical surface strings) or
``per_case_hierarchical_ner_pairs`` (hand-picked A×B pairs), this script builds a **full inventory**
per side and summary **A vs B** statistics in SNOMED concept space for each strict twin (same
``case_*.txt`` name).

Outputs
-------
1. ``per_case_ab_snomed_summary.csv`` — one row per case (counts, overlaps, mean depths).
2. ``per_case_ab_snomed_entity_inventory.csv`` — long format: every NER type per side with
   mention count, concept id, depth, and mapping status.
3. ``per_case_ab_snomed_cross_concepts.csv`` (optional) — within each case, concept pairs
   (c_a, c_b) with c_a ≠ c_b and a proper is-a link, capped per case for audit / downstream seeds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
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
from .paths import (
    CORPUS_A,
    CORPUS_B,
    DEFAULT_RF2_ROOT,
    SNOMED_PER_CASE_AB_CROSS,
    SNOMED_PER_CASE_AB_INVENTORY,
    SNOMED_PER_CASE_AB_SUMMARY,
    ensure_results,
)
from .rf2 import (
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    is_proper_isa_ancestor,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)


def _map_entities(
    counts: Counter[str],
    exact_map: dict[str, str],
    rows: list,
    inv: dict,
    parents: dict[str, list[str]],
    depth_memo: dict[str, int | None],
) -> tuple[dict[str, tuple[str, int]], set[str]]:
    """entity_lower -> (concept_id, depth); return also set of mapped concept ids."""
    ent_to_concept_depth: dict[str, tuple[str, int]] = {}
    concepts: set[str] = set()
    for ent, _n in counts.items():
        el = ent.strip().lower()
        if not el:
            continue
        cid = best_snomed_match(el, exact_map, rows, inv)
        if not cid:
            continue
        d = max_depth_from_root_rf2(cid, parents, RF2_ROOT_CONCEPT_ID, depth_memo)
        if d is None:
            continue
        ent_to_concept_depth[el] = (cid, int(d))
        concepts.add(cid)
    return ent_to_concept_depth, concepts


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Per strict case twin: all NER strings on A and on B → SNOMED; write summary + "
            "inventory + optional cross-concept is-a pairs for A vs B comparison."
        )
    )
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=f"Default: {SNOMED_PER_CASE_AB_SUMMARY.name} under results/snomed/.",
    )
    p.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help=f"Default: {SNOMED_PER_CASE_AB_INVENTORY.name} under results/snomed/.",
    )
    p.add_argument(
        "--no-cross-concepts",
        action="store_true",
        help="Skip writing cross-concept is-a pairs CSV.",
    )
    p.add_argument(
        "--cross-csv",
        type=Path,
        default=None,
        help=f"Default: {SNOMED_PER_CASE_AB_CROSS.name} under results/snomed/.",
    )
    p.add_argument(
        "--max-cross-pairs-per-case",
        type=int,
        default=40,
        metavar="K",
        help="Cap is-a concept pairs per case (sorted by |depth_a-depth_b| descending).",
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

    root = Path(__file__).resolve().parent.parent
    summary_path = args.summary_csv or (root / "results/snomed/per_case_ab_snomed_summary.csv")
    inv_path = args.inventory_csv or (root / "results/snomed/per_case_ab_snomed_entity_inventory.csv")
    cross_path = args.cross_csv or (root / "results/snomed/per_case_ab_snomed_cross_concepts.csv")

    print("Loading en_core_sci_lg …", file=sys.stderr)
    nlp = _load_nlp()

    summary_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    max_cross = max(0, args.max_cross_pairs_per_case)

    for path_a in sorted(args.corpus_a.glob("case_*.txt")):
        path_b = args.corpus_b / path_a.name
        if not path_b.is_file():
            continue
        stem = path_a.name
        try:
            text_a = path_a.read_text(encoding="utf-8", errors="replace")
            text_b = path_b.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        counts_a = collect_ner_entities_for_document(text_a, nlp)
        counts_b = collect_ner_entities_for_document(text_b, nlp)

        map_a, concepts_a = _map_entities(counts_a, exact_map, rows, inv, parents, depth_memo)
        map_b, concepts_b = _map_entities(counts_b, exact_map, rows, inv, parents, depth_memo)

        union_concepts = concepts_a | concepts_b
        inter = concepts_a & concepts_b
        only_a = concepts_a - concepts_b
        only_b = concepts_b - concepts_a

        strings_a = set(counts_a.keys())
        strings_b = set(counts_b.keys())
        string_inter = strings_a & strings_b

        def weighted_mean_depth(side: str) -> float | None:
            mp = map_a if side == "A" else map_b
            cnt = counts_a if side == "A" else counts_b
            num = 0.0
            den = 0
            for ent, k in cnt.items():
                el = ent.strip().lower()
                t = mp.get(el)
                if not t:
                    continue
                _cid, d = t
                num += d * k
                den += k
            return None if den == 0 else num / den

        mean_d_a = weighted_mean_depth("A")
        mean_d_b = weighted_mean_depth("B")

        jacc = len(inter) / len(union_concepts) if union_concepts else ""

        summary_rows.append(
            {
                "case_file": stem,
                "n_ner_types_a": len(counts_a),
                "n_ner_types_b": len(counts_b),
                "n_ner_mentions_a": int(sum(counts_a.values())),
                "n_ner_mentions_b": int(sum(counts_b.values())),
                "n_mapped_types_a": len(map_a),
                "n_mapped_types_b": len(map_b),
                "n_concepts_a": len(concepts_a),
                "n_concepts_b": len(concepts_b),
                "n_concepts_intersection": len(inter),
                "n_concepts_a_only": len(only_a),
                "n_concepts_b_only": len(only_b),
                "n_string_surface_intersection": len(string_inter),
                "jaccard_concepts_a_b": round(jacc, 6) if isinstance(jacc, float) else jacc,
                "mean_depth_weighted_a": "" if mean_d_a is None else round(mean_d_a, 4),
                "mean_depth_weighted_b": "" if mean_d_b is None else round(mean_d_b, 4),
                "mean_depth_weighted_b_minus_a": ""
                if mean_d_a is None or mean_d_b is None
                else round(mean_d_b - mean_d_a, 4),
            }
        )

        for side, counts, mp in (("A", counts_a, map_a), ("B", counts_b, map_b)):
            folder = args.corpus_a if side == "A" else args.corpus_b
            for ent, n in sorted(counts.items(), key=lambda x: (-x[1], x[0].lower())):
                el = ent.strip().lower()
                ctx = context_from_case_file(folder, stem, ent)
                hit = mp.get(el)
                if hit:
                    cid, d = hit
                    inventory_rows.append(
                        {
                            "case_file": stem,
                            "corpus_side": side,
                            "entity": ent,
                            "mention_count": int(n),
                            "snomed_concept": cid,
                            "depth": int(d),
                            "mapping_status": "mapped",
                            "context_sentence": (ctx or "")[:1200],
                        }
                    )
                else:
                    inventory_rows.append(
                        {
                            "case_file": stem,
                            "corpus_side": side,
                            "entity": ent,
                            "mention_count": int(n),
                            "snomed_concept": "",
                            "depth": "",
                            "mapping_status": "unmapped",
                            "context_sentence": (ctx or "")[:1200],
                        }
                    )

        if not args.no_cross_concepts and max_cross > 0:
            by_c_a: dict[str, list[str]] = {}
            for el, (cid, _d) in map_a.items():
                by_c_a.setdefault(cid, []).append(el)
            by_c_b: dict[str, list[str]] = {}
            for el, (cid, _d) in map_b.items():
                by_c_b.setdefault(cid, []).append(el)

            list_a = [(cid, map_a[sorted(by_c_a[cid])[0]][1], sorted(by_c_a[cid])[0]) for cid in concepts_a]
            list_b = [(cid, map_b[sorted(by_c_b[cid])[0]][1], sorted(by_c_b[cid])[0]) for cid in concepts_b]

            cand: list[tuple[int, dict[str, object]]] = []
            for ca, da, ea in list_a:
                for cb, db, eb in list_b:
                    if ca == cb:
                        continue
                    gap = abs(da - db)
                    a_broader = is_proper_isa_ancestor(parents, ca, cb)
                    b_broader = is_proper_isa_ancestor(parents, cb, ca)
                    if not a_broader and not b_broader:
                        continue
                    rel = "human_concept_broader" if a_broader else "model_concept_broader"
                    cand.append(
                        (
                            gap,
                            {
                                "case_file": stem,
                                "snomed_concept_a": ca,
                                "snomed_concept_b": cb,
                                "depth_a": da,
                                "depth_b": db,
                                "depth_gap_abs": gap,
                                "isa_broader_side": rel,
                                "entity_example_a": ea,
                                "entity_example_b": eb,
                            },
                        )
                    )
            cand.sort(key=lambda t: (-t[0], t[1]["snomed_concept_a"], t[1]["snomed_concept_b"]))
            for _, row in cand[:max_cross]:
                cross_rows.append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sum_fields = (
        list(summary_rows[0].keys())
        if summary_rows
        else [
            "case_file",
            "n_ner_types_a",
            "n_ner_types_b",
            "n_ner_mentions_a",
            "n_ner_mentions_b",
            "n_mapped_types_a",
            "n_mapped_types_b",
            "n_concepts_a",
            "n_concepts_b",
            "n_concepts_intersection",
            "n_concepts_a_only",
            "n_concepts_b_only",
            "n_string_surface_intersection",
            "jaccard_concepts_a_b",
            "mean_depth_weighted_a",
            "mean_depth_weighted_b",
            "mean_depth_weighted_b_minus_a",
        ]
    )
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in sum_fields})

    inv_fields = [
        "case_file",
        "corpus_side",
        "entity",
        "mention_count",
        "snomed_concept",
        "depth",
        "mapping_status",
        "context_sentence",
    ]
    with inv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields, extrasaction="ignore")
        w.writeheader()
        for r in inventory_rows:
            w.writerow(r)

    cross_fields = [
        "case_file",
        "snomed_concept_a",
        "snomed_concept_b",
        "depth_a",
        "depth_b",
        "depth_gap_abs",
        "isa_broader_side",
        "entity_example_a",
        "entity_example_b",
    ]
    if not args.no_cross_concepts:
        with cross_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cross_fields, extrasaction="ignore")
            w.writeheader()
            for r in cross_rows:
                w.writerow(r)

    print(f"Wrote {len(summary_rows)} summary row(s) → {summary_path}", file=sys.stderr)
    print(f"Wrote {len(inventory_rows)} inventory row(s) → {inv_path}", file=sys.stderr)
    if args.no_cross_concepts:
        print("Cross-concept CSV skipped (--no-cross-concepts).", file=sys.stderr)
    else:
        print(f"Wrote {len(cross_rows)} cross-concept row(s) → {cross_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
