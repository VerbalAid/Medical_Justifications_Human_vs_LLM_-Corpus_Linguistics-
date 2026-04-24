#!/usr/bin/env python3
"""
Metrics on a fixed case list: hedge rate /1k, 'most likely' /1k, mean SNOMED depth on shared NER entities.

Requires en_core_sci_lg and RF2 (same as ner_depth). Corpus folders should each contain only the
listed case_*.txt files (or superset); only listed stems are read.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .hedging import TOKEN_RE, hedge_hits_and_tokens
from .ner_depth import (
    _ensure_python,
    _load_nlp,
    best_snomed_match,
    collect_ner_entities,
    index_description_rows,
    load_rf2_en_description_rows,
    load_rf2_term_to_concept,
)
from .paths import DEFAULT_RF2_ROOT, ROOT, ensure_results
from .rf2 import (
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)

MOST_LIKELY_RE = re.compile(r"\bmost\s+likely\b", re.IGNORECASE)


def load_stems(path: Path) -> list[str]:
    stems: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.lower().endswith(".txt"):
            line = f"case_{int(line):03d}.txt"
        stems.append(line)
    return stems


def merged_text(folder: Path, stems: list[str]) -> str:
    parts: list[str] = []
    for s in stems:
        p = folder / s
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def hedge_per_1k(text: str) -> float:
    h, t = hedge_hits_and_tokens(text)
    return (1000.0 * h / t) if t else 0.0


def most_likely_per_1k(text: str) -> float:
    t = sum(1 for _ in TOKEN_RE.finditer(text))
    c = len(MOST_LIKELY_RE.findall(text))
    return (1000.0 * c / t) if t else 0.0


def entity_depth_map(
    counter: Counter[str],
    exact_map: dict,
    rows: list,
    inv: dict,
    isa_parents: dict,
    depth_memo: dict,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for ent, _freq in counter.items():
        cid = best_snomed_match(ent, exact_map, rows, inv)
        if not cid:
            continue
        d = max_depth_from_root_rf2(cid, isa_parents, memo=depth_memo)
        if d is not None:
            out[ent.lower()] = d
    return out


def triple_shared_depths(
    map_a: dict[str, int],
    map_k: dict[str, int],
    map_u: dict[str, int],
) -> tuple[float | None, float | None, float | None, int]:
    """Mean depth per corpus on entities that have a resolvable depth in all three."""
    triple = [e for e in map_a if e in map_k and e in map_u]
    if not triple:
        return None, None, None, 0
    ma = sum(map_a[e] for e in triple) / len(triple)
    mk = sum(map_k[e] for e in triple) / len(triple)
    mu = sum(map_u[e] for e in triple) / len(triple)
    return ma, mk, mu, len(triple)


def main() -> None:
    _ensure_python()
    ensure_results()
    p = argparse.ArgumentParser(description="Ablation stats: hedge, most likely, mean SNOMED depth (shared entities).")
    p.add_argument("--cases", type=Path, default=ROOT / "cases_unkeyed40.txt")
    p.add_argument("--corpus-a", type=Path, default=ROOT / "corpus_a")
    p.add_argument("--corpus-b-keyed", type=Path, default=ROOT / "corpus_b")
    p.add_argument("--corpus-b-unkeyed", type=Path, default=ROOT / "corpus_b_unkeyed")
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument("--out-json", type=Path, default=ROOT / "results" / "ablation_unkeyed40_metrics.json")
    args = p.parse_args()

    stems = load_stems(args.cases)
    if len(stems) != 40:
        print(f"Warning: expected 40 stems, got {len(stems)}", file=sys.stderr)

    ta = merged_text(args.corpus_a, stems)
    tk = merged_text(args.corpus_b_keyed, stems)
    tu = merged_text(args.corpus_b_unkeyed, stems)

    ha, hk, hu = hedge_per_1k(ta), hedge_per_1k(tk), hedge_per_1k(tu)
    ma, mk, mu = most_likely_per_1k(ta), most_likely_per_1k(tk), most_likely_per_1k(tu)

    rf2 = args.rf2_root.resolve()
    if not rf2.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2))
    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    print("Loading spaCy …", file=sys.stderr)
    nlp = _load_nlp()
    print("Copying subset into temp dirs for NER …", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="ablation_unkeyed_") as tmp:
        da_dir = Path(tmp) / "a"
        dk_dir = Path(tmp) / "bk"
        du_dir = Path(tmp) / "bu"
        for d in (da_dir, dk_dir, du_dir):
            d.mkdir()
        for s in stems:
            for src_root, dst in (
                (args.corpus_a, da_dir),
                (args.corpus_b_keyed, dk_dir),
                (args.corpus_b_unkeyed, du_dir),
            ):
                sp = src_root / s
                if sp.is_file():
                    shutil.copy2(sp, dst / s)

        cnt_a = collect_ner_entities(da_dir, nlp)
        cnt_k = collect_ner_entities(dk_dir, nlp)
        cnt_u = collect_ner_entities(du_dir, nlp)

    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    inv = index_description_rows(rows)
    isa_parents = load_rf2_isa_parent_map(rel_files)
    depth_memo: dict[str, int | None] = {}

    map_a = entity_depth_map(cnt_a, exact_map, rows, inv, isa_parents, depth_memo)
    map_k = entity_depth_map(cnt_k, exact_map, rows, inv, isa_parents, depth_memo)
    map_u = entity_depth_map(cnt_u, exact_map, rows, inv, isa_parents, depth_memo)

    d_a, d_k, d_u, n_triple = triple_shared_depths(map_a, map_k, map_u)
    snomed_delta = (d_u - d_k) if d_u is not None and d_k is not None else None

    out = {
        "n_cases": len(stems),
        "hedge_per_1000": {"A": ha, "B_keyed": hk, "B_unkeyed": hu, "delta_unkeyed_minus_keyed": hu - hk},
        "most_likely_per_1000": {"A": ma, "B_keyed": mk, "B_unkeyed": mu, "delta_unkeyed_minus_keyed": mu - mk},
        "mean_snomed_depth_shared_entities": {
            "A_on_triple": d_a,
            "B_keyed_on_triple": d_k,
            "B_unkeyed_on_triple": d_u,
            "delta_unkeyed_minus_keyed_B": snomed_delta,
            "n_entities_triple_intersection": n_triple,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
