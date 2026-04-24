#!/usr/bin/env python3
# JPEG taxonomy figures from llm_aligned_snomed_depths.csv (same layout as snomed_taxonomy_compare).

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .branch_pairs import ancestor_closure
from .corpus_context import context_first_file_containing, context_from_case_file
from .paths import (
    CORPUS_A,
    CORPUS_B,
    DEFAULT_RF2_ROOT,
    DIR_LLM_TAXONOMY_JPEG,
    LLM_ALIGNED_DEPTHS_CSV,
    LLM_TAXONOMY_OVERVIEW,
    ensure_results,
)
from .rf2 import (
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)
from .snomed_taxonomy_compare import (
    category_slug,
    compute_pair_visual,
    load_rf2_labels,
    shortest_path_up,
    write_category_jpeg_figure,
    write_overview_jpeg,
)


def _parse_int(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def deepest_common_ancestor(
    ca: str,
    cb: str,
    child_to_parents: dict[str, list[str]],
    depth_memo: dict[str, int | None],
) -> str | None:
    """Same rule as branch_pairs: common ancestor with greatest depth from root."""
    anc_a = ancestor_closure(ca, child_to_parents)
    anc_b = ancestor_closure(cb, child_to_parents)
    if len(anc_a) <= len(anc_b):
        common = [x for x in anc_a if x in anc_b]
    else:
        common = [x for x in anc_b if x in anc_a]
    if not common:
        return None
    best: str | None = None
    best_d = -1
    for x in common:
        dx = max_depth_from_root_rf2(x, child_to_parents, RF2_ROOT_CONCEPT_ID, depth_memo)
        if dx is not None and dx > best_d:
            best_d = dx
            best = x
    return best


def llm_row_to_branch_style(
    row: dict[str, str],
    child_to_parents: dict[str, list[str]],
    depth_memo: dict[str, int | None],
) -> dict[str, str] | None:
    ca = (row.get("snomed_concept_a") or "").strip()
    cb = (row.get("snomed_concept_b") or "").strip()
    da = _parse_int(row.get("depth_a", ""))
    db = _parse_int(row.get("depth_b", ""))
    if not ca or not cb:
        return None
    if da is None:
        da = max_depth_from_root_rf2(ca, child_to_parents, RF2_ROOT_CONCEPT_ID, depth_memo)
    if db is None:
        db = max_depth_from_root_rf2(cb, child_to_parents, RF2_ROOT_CONCEPT_ID, depth_memo)
    if da is None or db is None:
        return None
    if ca == cb:
        return None
    lca = deepest_common_ancestor(ca, cb, child_to_parents, depth_memo)
    if not lca:
        return None
    ea = (row.get("final_entity_a") or row.get("orig_entity_a") or "").strip()
    eb = (row.get("final_entity_b") or row.get("orig_entity_b") or "").strip()
    cat = (row.get("category") or "Unclassified").strip()
    dec = (row.get("llm_decision") or "").strip()
    return {
        "concept_a": ca,
        "concept_b": cb,
        "lca_concept": lca,
        "depth_a": str(da),
        "depth_b": str(db),
        "entity_a": ea,
        "entity_b": eb,
        "_category": cat,
        "_llm_decision": dec,
    }


def context_for_llm_depth_row(row: dict[str, str]) -> tuple[str, str]:
    """Prefer CSV columns; otherwise extract a sentence around the original NER strings."""
    ca = (row.get("context_sentence_a") or "").strip()
    cb = (row.get("context_sentence_b") or "").strip()
    if ca and cb:
        return ca, cb
    stem = (row.get("case_stem") or "").strip()
    oa = (row.get("orig_entity_a") or "").strip()
    ob = (row.get("orig_entity_b") or "").strip()
    if stem and "|" not in stem:
        if not ca:
            ca = context_from_case_file(CORPUS_A, stem, oa)
        if not cb:
            cb = context_from_case_file(CORPUS_B, stem, ob)
    if not ca:
        ca = context_first_file_containing(CORPUS_A, oa)
    if not cb:
        cb = context_first_file_containing(CORPUS_B, ob)
    return ca, cb


def depth_csv_row_to_overview_row(row: dict[str, str]) -> dict[str, str] | None:
    """Minimal row dict for write_overview_jpeg (compact layout); no SNOMED path required."""
    fa = (row.get("final_entity_a") or row.get("orig_entity_a") or "").strip()
    fb = (row.get("final_entity_b") or row.get("orig_entity_b") or "").strip()
    if not fa or not fb:
        return None
    da = _parse_int(row.get("depth_a", ""))
    db = _parse_int(row.get("depth_b", ""))
    if da is None or db is None:
        return None
    ctx_a, ctx_b = context_for_llm_depth_row(row)
    if not (ctx_a or ctx_b):
        return None
    stem = (row.get("case_stem") or "").strip()
    cat = (row.get("category") or "").strip()
    title = stem if stem else (cat[:48] if cat else "row")
    return {
        "entity_a": fa,
        "entity_b": fb,
        "depth_a": str(da),
        "depth_b": str(db),
        "_llm_decision": (row.get("llm_decision") or "").strip(),
        "_category": title,
        "context_a": ctx_a or "",
        "context_b": ctx_b or "",
    }


def _is_per_case_depth_csv(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return "per-case" in (rows[0].get("category") or "").lower()


def _decision_upper_row(r: dict[str, str]) -> str:
    return (r.get("_llm_decision") or "").strip().upper()


def _pick_overview_by_decision(
    pool: list[dict[str, str]],
    decision: str,
    limit: int,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    want = decision.strip().upper()
    out: list[dict[str, str]] = []
    for r in pool:
        if _decision_upper_row(r) != want:
            continue
        k = (r.get("_category") or "", r.get("entity_a") or "", r.get("entity_b") or "")
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def collect_label_ids(pairs: list[dict[str, str]], parents: dict[str, list[str]]) -> set[str]:
    want: set[str] = set()
    for r in pairs:
        for k in ("concept_a", "concept_b", "lca_concept"):
            want.add(r[k].strip())
        for ca_s, lca_s in ((r["concept_a"], r["lca_concept"]), (r["concept_b"], r["lca_concept"])):
            pa = shortest_path_up(ca_s.strip(), lca_s.strip(), parents)
            if pa:
                want.update(pa)
    return want


def main() -> None:
    ensure_results()
    p = argparse.ArgumentParser(
        description=(
            "Build per-category + overview JPEGs from llm_aligned_snomed_depths.csv "
            "(LLM-final strings and depths), using the same figure style as snomed_taxonomy_compare."
        )
    )
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument("--align-csv", type=Path, default=LLM_ALIGNED_DEPTHS_CSV)
    p.add_argument("--jpeg-dir", type=Path, default=DIR_LLM_TAXONOMY_JPEG)
    p.add_argument("--overview-jpeg", type=Path, default=LLM_TAXONOMY_OVERVIEW)
    p.add_argument("--no-overview-jpeg", action="store_true")
    p.add_argument("--no-category-jpeg", action="store_true")
    p.add_argument(
        "--overview-comparable",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Overview strip: up to N SUBSTITUTE panels first. "
            "Per-case CSV: chosen from the full depth CSV. Branch CSV: drawable RF2 rows only."
        ),
    )
    p.add_argument(
        "--overview-contrast",
        type=int,
        default=2,
        metavar="M",
        help=(
            "Overview strip: after substitutes, up to M INCOMPARABLE panels. "
            "Per-case: full CSV; branch: drawable rows only."
        ),
    )
    p.add_argument(
        "--overview-keep",
        type=int,
        default=3,
        metavar="K",
        help="Per-case overview only: after SUBSTITUTE and INCOMPARABLE, up to K KEEP panels.",
    )
    p.add_argument(
        "--overview-max",
        type=int,
        default=None,
        metavar="K",
        help="Optional hard cap on overview panels after substitute+contrast+keep selection (None = no extra cap).",
    )
    p.add_argument(
        "--no-overview-intro",
        action="store_true",
        help="Omit the boxed explanatory header on the overview JPEG.",
    )
    args = p.parse_args()

    if not args.align_csv.is_file():
        raise SystemExit(
            f"Missing {args.align_csv}\n"
            "Run: python -m pipeline.llm_pair_align\n"
            "Then: python -m pipeline.llm_pair_align --depth-only\n"
        )

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    print("Loading is-a map …", file=sys.stderr)
    parents = load_rf2_isa_parent_map(rel_files)
    depth_memo: dict[str, int | None] = {}

    raw_rows: list[dict[str, str]] = []
    with args.align_csv.open(encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    pairs: list[dict[str, str]] = []
    skipped = 0
    for row in raw_rows:
        conv = llm_row_to_branch_style(row, parents, depth_memo)
        if conv is None:
            skipped += 1
            continue
        ctx_a, ctx_b = context_for_llm_depth_row(row)
        if ctx_a:
            conv["context_a"] = ctx_a
        if ctx_b:
            conv["context_b"] = ctx_b
        pairs.append(conv)

    per_case = _is_per_case_depth_csv(raw_rows)
    if not pairs:
        if per_case and raw_rows:
            print(
                f"No drawable taxonomy rows in {args.align_csv} (skipped {skipped}). "
                "Category JPEGs skipped; overview can still use all depth-annotated rows.",
                file=sys.stderr,
            )
        else:
            raise SystemExit(
                f"No drawable rows in {args.align_csv} "
                "(need two SNOMED concept IDs, resolvable is-a depths, and a non-trivial LCA path). "
                f"Skipped {skipped} row(s)."
            )
    else:
        print(f"Drawable LLM-aligned pairs: {len(pairs)} (skipped {skipped}).", file=sys.stderr)

    want = collect_label_ids(pairs, parents) if pairs else set()
    print("Loading EN labels …", file=sys.stderr)
    labels = load_rf2_labels(desc_files, want)

    args.jpeg_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, r in enumerate(pairs, start=1):
        slug = category_slug(r.get("_category", "category"))
        vis = compute_pair_visual(r, labels, parents)
        if vis is None:
            print(f"Skip figure (no path to LCA): {r.get('_category')}", file=sys.stderr)
            continue
        dec = r.get("_llm_decision", "").strip()
        if dec:
            vis["title"] = f"{vis['title']} — LLM: {dec}"
        if r.get("context_a"):
            vis["context_a"] = str(r["context_a"])
        if r.get("context_b"):
            vis["context_b"] = str(r["context_b"])
        if not args.no_category_jpeg:
            jpath = args.jpeg_dir / f"llm_taxonomy_{i:02d}_{slug}.jpg"
            write_category_jpeg_figure(vis, labels, jpath)
            written.append(jpath)

    for path in written:
        print(path, file=sys.stderr)
    print(f"Wrote {len(written)} JPEG(s) under {args.jpeg_dir}", file=sys.stderr)

    if not args.no_overview_jpeg:
        overview_pool: list[dict[str, str]] = []
        for row in raw_rows:
            od = depth_csv_row_to_overview_row(row)
            if od:
                overview_pool.append(od)

        overview_rows: list[dict[str, str]] = []
        if per_case and overview_pool:
            seen_keys: set[tuple[str, str, str]] = set()
            overview_rows.extend(
                _pick_overview_by_decision(
                    overview_pool, "SUBSTITUTE", max(0, args.overview_comparable), seen_keys
                )
            )
            overview_rows.extend(
                _pick_overview_by_decision(
                    overview_pool, "INCOMPARABLE", max(0, args.overview_contrast), seen_keys
                )
            )
            overview_rows.extend(
                _pick_overview_by_decision(overview_pool, "KEEP", max(0, args.overview_keep), seen_keys)
            )
        else:

            def _decision_upper_pair(r: dict[str, str]) -> str:
                return (r.get("_llm_decision") or "").strip().upper()

            subst = [r for r in pairs if _decision_upper_pair(r) == "SUBSTITUTE"]
            subst.sort(key=lambda r: (r.get("_category") or "").lower())
            inc = [r for r in pairs if _decision_upper_pair(r) == "INCOMPARABLE"]
            inc.sort(
                key=lambda r: (
                    abs(int(r["depth_a"]) - int(r["depth_b"])),
                    (r.get("_category") or "").lower(),
                )
            )

            overview_rows.extend(subst[: max(0, args.overview_comparable)])
            taken = {id(x) for x in overview_rows}
            n_contrast = max(0, args.overview_contrast)
            for r in inc:
                if n_contrast <= 0:
                    break
                if id(r) in taken:
                    continue
                overview_rows.append(r)
                taken.add(id(r))
                n_contrast -= 1

        if args.overview_max is not None and args.overview_max > 0:
            overview_rows = overview_rows[: args.overview_max]

        if not overview_rows:
            print("Overview skipped: no rows with entities, depths, and context.", file=sys.stderr)
        else:
            intro_lines = None
            if not args.no_overview_intro:
                if per_case:
                    intro_lines = [
                        (
                            "Per-case same-string audit: for each case file we pair the same highlighted surface string "
                            "in the human (A) and model (B) sessions; Mistral reviews one row per pair (JSON)."
                        ),
                        "Depth values are RF2 is-a distance from the SNOMED CT root for the reviewer-chosen concepts.",
                        "KEEP — reviewer accepts that both spans pick out the same clinical referent for that token.",
                        (
                            "SUBSTITUTE — reviewer replaces one or both surface strings so SNOMED depths compare "
                            "the concepts they intend (fairest rows when the string match is misleading)."
                        ),
                        (
                            "INCOMPARABLE — reviewer declines a single clinical pairing across the two spans "
                            "(often different angles on the vignette); not an indicator of broken extraction or invalid SNOMED IDs."
                        ),
                    ]
                else:
                    intro_lines = [
                        (
                            "Divergent seeds: one Human (A) vs Model (B) highlight per SNOMED top-level category, "
                            "then one Mistral JSON review per row. Depths are RF2 is-a distance from the SNOMED root."
                        ),
                        (
                            "SUBSTITUTE — the reviewer replaces the surface strings so the two sides compare intended "
                            "clinical concepts; these are the fairest rows for a depth read-off."
                        ),
                        (
                            "INCOMPARABLE — the reviewer refuses to equate the two highlighted spans as one clinical pairing "
                            "(often different angles on the case). It is not a mark of broken extraction or invalid SNOMED codes."
                        ),
                    ]
            write_overview_jpeg(overview_rows, args.overview_jpeg, intro_lines=intro_lines)


if __name__ == "__main__":
    main()
