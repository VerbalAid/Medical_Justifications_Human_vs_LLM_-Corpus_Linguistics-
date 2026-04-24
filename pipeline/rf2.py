#!/usr/bin/env python3
# Helpers to read SNOMED RF2: word to concept id, is-a parents, depth from root. Also a small CLI.

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from .paths import DEFAULT_RF2_ROOT, FREQ_TOP_HUMAN, FREQ_TOP_MODEL, SNOMED_DEPTH_FREQ, ensure_results

RF2_ISA_TYPE_ID = "116680003"
RF2_ROOT_CONCEPT_ID = "138875005"
RF2_FSN_TYPE_ID = "900000000000003001"
RF2_SYNONYM_TYPE_ID = "900000000000013009"


def _csv_field_limit() -> None:
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def format_rf2_root_missing_error(rf2_root: Path) -> str:
    lines = [
        f"RF2 root does not exist: {rf2_root}",
        "",
        "Unpack the SNOMED CT International RF2 release here, or pass --rf2-root to the folder "
        "that contains Snapshot/ or Full/ with sct2_Description_*.txt and sct2_Relationship_*.txt.",
    ]
    s = str(rf2_root)
    if "path/to" in s.lower() or "/path/" in s or "..." in s:
        lines += [
            "",
            "That path looks like a placeholder.",
            f"Default next to the project is: {DEFAULT_RF2_ROOT}",
        ]
    return "\n".join(lines)


def discover_rf2_description_and_relationship_files(
    rf2_root: Path,
) -> tuple[list[Path], list[Path]]:
    desc: list[Path] = []
    rel: list[Path] = []
    for p in sorted(rf2_root.glob("**/*.txt")):
        if not p.is_file():
            continue
        n = p.name.lower()
        if not n.startswith("sct2_"):
            continue
        if n.startswith("sct2_description"):
            desc.append(p)
        elif "relationship" in n and "concrete" not in n:
            rel.append(p)
    return desc, rel


def load_rf2_term_to_concept(description_paths: list[Path]) -> dict[str, str]:
    """Lowercased term -> concept id (FSN preferred over synonym)."""
    best: dict[str, tuple[str, int]] = {}
    rank_fsn, rank_syn = 0, 1
    _csv_field_limit()
    for path in description_paths:
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                continue
            col = {name: i for i, name in enumerate(header)}
            need = ("conceptId", "languageCode", "typeId", "term", "active")
            if not all(k in col for k in need):
                print(f"  [skip] {path.name}: missing columns {need}", file=sys.stderr)
                continue
            ci, lc, ti, te, ac = (col[k] for k in need)
            for row in reader:
                if max(ci, lc, ti, te, ac) >= len(row):
                    continue
                if row[ac] != "1":
                    continue
                if not row[lc].strip().lower().startswith("en"):
                    continue
                typ = row[ti]
                if typ not in (RF2_FSN_TYPE_ID, RF2_SYNONYM_TYPE_ID):
                    continue
                key = row[te].strip().lower()
                if not key:
                    continue
                cid = row[ci]
                rank = rank_fsn if typ == RF2_FSN_TYPE_ID else rank_syn
                prev = best.get(key)
                if prev is None or rank < prev[1]:
                    best[key] = (cid, rank)
    return {k: v[0] for k, v in best.items()}


def load_rf2_en_description_rows(description_paths: list[Path]) -> list[tuple[str, str, int]]:
    """(conceptId, term_lower, type_rank) for active EN FSN/synonym rows."""
    rows: list[tuple[str, str, int]] = []
    rank_fsn, rank_syn = 0, 1
    _csv_field_limit()
    for path in description_paths:
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                continue
            col = {name: i for i, name in enumerate(header)}
            need = ("conceptId", "languageCode", "typeId", "term", "active")
            if not all(k in col for k in need):
                continue
            ci, lc, ti, te, ac = (col[k] for k in need)
            for row in reader:
                if max(ci, lc, ti, te, ac) >= len(row):
                    continue
                if row[ac] != "1" or not row[lc].strip().lower().startswith("en"):
                    continue
                typ = row[ti]
                if typ not in (RF2_FSN_TYPE_ID, RF2_SYNONYM_TYPE_ID):
                    continue
                key = row[te].strip().lower()
                if not key:
                    continue
                rank = rank_fsn if typ == RF2_FSN_TYPE_ID else rank_syn
                rows.append((row[ci], key, rank))
    return rows


def load_rf2_isa_parent_map(relationship_paths: list[Path]) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {}
    _csv_field_limit()
    for path in relationship_paths:
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                continue
            col = {name: i for i, name in enumerate(header)}
            need = ("sourceId", "destinationId", "typeId", "active")
            if not all(k in col for k in need):
                print(f"  [skip] {path.name}: missing columns {need}", file=sys.stderr)
                continue
            si, di, ti, ac = (col[k] for k in need)
            for row in reader:
                if max(si, di, ti, ac) >= len(row):
                    continue
                if row[ac] != "1" or row[ti] != RF2_ISA_TYPE_ID:
                    continue
                parents.setdefault(row[si], []).append(row[di])
    return parents


def is_proper_isa_ancestor(
    child_to_parents: dict[str, list[str]],
    ancestor_id: str,
    descendant_id: str,
    *,
    max_hops: int = 120,
) -> bool:
    """True iff ``descendant_id`` can reach ``ancestor_id`` by walking >=1 active is-a parent links.

    Used to pair a broader SNOMED concept with a strictly more specific descendant in the same
    hierarchy (distinct concepts).
    """
    if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
        return False
    frontier = list(child_to_parents.get(descendant_id, []))
    seen: set[str] = set()
    hops = 0
    while frontier and hops < max_hops:
        hops += 1
        nxt: list[str] = []
        for p in frontier:
            if p == ancestor_id:
                return True
            if p in seen:
                continue
            seen.add(p)
            nxt.extend(child_to_parents.get(p, []))
        frontier = nxt
    return False


def max_depth_from_root_rf2(
    concept_id: str,
    child_to_parents: dict[str, list[str]],
    root_id: str = RF2_ROOT_CONCEPT_ID,
    memo: dict[str, int | None] | None = None,
    stack: set[str] | None = None,
) -> int | None:
    if memo is None:
        memo = {}
    if stack is None:
        stack = set()
    if concept_id == root_id:
        return 0
    if concept_id in memo:
        return memo[concept_id]
    if concept_id in stack:
        memo[concept_id] = None
        return None
    pars = child_to_parents.get(concept_id)
    if not pars:
        memo[concept_id] = None
        return None
    stack.add(concept_id)
    best: int | None = None
    for p in pars:
        d = max_depth_from_root_rf2(p, child_to_parents, root_id, memo, stack)
        if d is not None:
            cand = 1 + d
            best = cand if best is None else max(best, cand)
    stack.discard(concept_id)
    memo[concept_id] = best
    return best


def load_top_terms(freq_csv: Path, k: int) -> list[str]:
    terms: list[str] = []
    with freq_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if len(terms) >= k:
                break
            terms.append(row["term"].strip())
    return terms


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(
        description="Top frequency terms -> SNOMED concept + is-a depth (RF2)."
    )
    parser.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    parser.add_argument("--freq-a", type=Path, default=FREQ_TOP_HUMAN)
    parser.add_argument("--freq-b", type=Path, default=FREQ_TOP_MODEL)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, default=SNOMED_DEPTH_FREQ)
    args = parser.parse_args()

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))
    if not rf2_root.is_dir():
        raise SystemExit(f"RF2 root must be a directory: {rf2_root}")

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    print(f"RF2 root: {rf2_root}")
    print(f"  {len(desc_files)} Description *.txt, {len(rel_files)} Relationship *.txt")
    if not desc_files or not rel_files:
        raise SystemExit("Need both Description and Relationship .txt under RF2 root.")

    print(f"  Example Description: {desc_files[0]}")
    print(f"  Example Relationship: {rel_files[0]}")

    print("  Loading EN descriptions -> conceptId …")
    term_to_concept = load_rf2_term_to_concept(desc_files)
    print(f"  {len(term_to_concept):,} distinct strings (lowercased).")

    print("  Loading is-a …")
    isa_parents = load_rf2_isa_parent_map(rel_files)

    terms_a = load_top_terms(args.freq_a, args.top)
    terms_b = load_top_terms(args.freq_b, args.top)
    tasks = [(t, "A") for t in terms_a] + [(t, "B") for t in terms_b]

    out_rows: list[dict[str, str | int]] = []
    matched_concept = matched_depth = 0
    rf2_memo: dict[str, int | None] = {}

    for term, corpus in tasks:
        snomed_concept = ""
        depth_val: int | str = ""
        scui = term_to_concept.get(term.strip().lower())
        if scui:
            matched_concept += 1
            snomed_concept = scui
            d = max_depth_from_root_rf2(scui, isa_parents, memo=rf2_memo)
            if d is not None:
                depth_val = d
                matched_depth += 1
        out_rows.append(
            {"term": term, "corpus": corpus, "snomed_concept": snomed_concept, "depth": depth_val}
        )

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "corpus", "snomed_concept", "depth"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    n = len(tasks)
    print(f"Wrote {args.out} ({n} rows).")
    print(
        f"Coverage: concept {matched_concept}/{n} = {matched_concept / n if n else 0:.3f}; "
        f"depth {matched_depth}/{n} = {matched_depth / n if n else 0:.3f}"
    )


if __name__ == "__main__":
    main()
