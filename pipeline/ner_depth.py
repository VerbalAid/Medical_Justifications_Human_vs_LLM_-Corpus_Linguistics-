#!/usr/bin/env python3
# NER on each case file (scispaCy), match entity text to SNOMED, write a depth CSV. Best on Python 3.12.

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import Counter

from .paths import CORPUS_A, CORPUS_B, DEFAULT_RF2_ROOT, SNOMED_DEPTH_FREQ, SNOMED_DEPTH_NER, ensure_results
from .rf2 import (
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_en_description_rows,
    load_rf2_isa_parent_map,
    load_rf2_term_to_concept,
    max_depth_from_root_rf2,
)

OUT_CSV = SNOMED_DEPTH_NER
LEGACY_DEPTH = SNOMED_DEPTH_FREQ

SCI_LG_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
    "en_core_sci_lg-0.5.4.tar.gz"
)

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _install_hint() -> str:
    return (
        "Use Python 3.12 for spaCy (Fedora’s default venv is often 3.14 and breaks wheels):\n\n"
        "  rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
        "  python -m pip install -U pip scispacy\n"
        f"  python -m pip install {SCI_LG_URL}\n\n"
        "If numpy builds from source: sudo dnf install python3.12-devel\n"
    )


def _ensure_python() -> None:
    if sys.version_info >= (3, 14):
        raise SystemExit(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported for spaCy.\n"
            f"Interpreter: {sys.executable}\n\n"
            "Until you use 3.12, `snomed_depth_from_ner_entities.csv` will not exist, so "
            "graph_viz / pair_depths / branch_pairs / snomed_taxonomy_compare will fail.\n\n"
            + _install_hint()
        )


def _load_nlp():
    try:
        import spacy
    except ImportError as e:
        raise SystemExit(f"Cannot import spaCy ({e}).\n{sys.executable}\n\n" + _install_hint()) from e
    try:
        return spacy.load("en_core_sci_lg")
    except OSError as e:
        raise SystemExit(f'Model en_core_sci_lg missing ({e}).\n\n' + _install_hint()) from e


def collect_ner_entities(folder, nlp) -> Counter[str]:
    counts: Counter[str] = Counter()
    paths = sorted(folder.glob("case_*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    for doc in nlp.pipe(texts, batch_size=8):
        for ent in doc.ents:
            t = ent.text.strip().lower()
            if t:
                counts[t] += 1
    return counts


def collect_ner_entities_for_document(text: str, nlp) -> Counter[str]:
    """Lowercased entity surface strings from one document (e.g. one case file)."""
    counts: Counter[str] = Counter()
    doc = nlp(text or "")
    for ent in doc.ents:
        t = ent.text.strip().lower()
        if t:
            counts[t] += 1
    return counts


def entity_in_description(entity_lower: str, desc_lower: str) -> bool:
    if entity_lower == desc_lower:
        return True
    if not entity_lower:
        return False
    if " " in entity_lower or "-" in entity_lower:
        return entity_lower in desc_lower
    if len(entity_lower) < 2:
        return False
    return (
        re.search(
            r"(?<![a-z0-9])" + re.escape(entity_lower) + r"(?![a-z0-9])",
            desc_lower,
        )
        is not None
    )


def index_description_rows(rows: list[tuple[str, str, int]]) -> dict[str, set[int]]:
    inv: dict[str, set[int]] = {}
    for i, (_cid, term_lower, _rank) in enumerate(rows):
        for w in WORD_RE.findall(term_lower):
            if len(w) >= 2:
                inv.setdefault(w.lower(), set()).add(i)
    return inv


def candidate_row_indices(entity_lower: str, inv: dict[str, set[int]]) -> set[int]:
    words = [w.lower() for w in WORD_RE.findall(entity_lower) if len(w) >= 2]
    if not words:
        return set()
    sets_list = [inv[w] for w in words if w in inv]
    if not sets_list:
        return set()
    sets_list.sort(key=len)
    inter = set(sets_list[0])
    for s in sets_list[1:]:
        inter &= s
    if inter:
        return inter
    uni: set[int] = set()
    for s in sets_list:
        uni |= s
    if len(uni) <= 250_000:
        return uni
    return sets_list[0]


def best_snomed_match(
    entity_lower: str,
    exact_map: dict[str, str],
    rows: list[tuple[str, str, int]],
    inv: dict[str, set[int]],
) -> str | None:
    if entity_lower in exact_map:
        return exact_map[entity_lower]
    cand_idx = candidate_row_indices(entity_lower, inv)
    best: tuple[str, int, int] | None = None
    for i in cand_idx:
        cid, term_lower, rank = rows[i]
        if not entity_in_description(entity_lower, term_lower):
            continue
        L = len(term_lower)
        if best is None or rank < best[1] or (rank == best[1] and L < best[2]):
            best = (cid, rank, L)
    return best[0] if best else None


def mean_depth_for_corpus(rows_out: list[dict], corpus: str) -> tuple[float, int]:
    depths = [int(r["depth"]) for r in rows_out if r["corpus"] == corpus and str(r["depth"]).isdigit()]
    if not depths:
        return 0.0, 0
    return sum(depths) / len(depths), len(depths)


def top_depth_extremes(rows_matched: list[dict], corpus: str, k: int) -> tuple[list[dict], list[dict]]:
    sub = [r for r in rows_matched if r["corpus"] == corpus and str(r["depth"]).isdigit()]
    sub.sort(key=lambda r: int(r["depth"]), reverse=True)
    deepest = sub[:k]
    shallowest = sorted(sub, key=lambda r: int(r["depth"]))[:k]
    return deepest, shallowest


def load_legacy_freq_depth_means(path) -> tuple[float | None, float | None, int, int]:
    if not path.is_file():
        return None, None, 0, 0
    da: list[int] = []
    db: list[int] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("depth", "")
            if not str(d).isdigit():
                continue
            v = int(d)
            if row.get("corpus") == "A":
                da.append(v)
            elif row.get("corpus") == "B":
                db.append(v)
    ma = sum(da) / len(da) if da else None
    mb = sum(db) / len(db) if db else None
    return ma, mb, len(da), len(db)


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(description="NER -> SNOMED RF2 -> is-a depth.")
    parser.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    args = parser.parse_args()
    _ensure_python()
    rf2_root = args.rf2_root.resolve()

    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))
    if not rf2_root.is_dir():
        raise SystemExit(f"RF2 root must be a directory: {rf2_root}")

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need Description and Relationship .txt under --rf2-root.")

    print("Loading en_core_sci_lg …")
    nlp = _load_nlp()

    print("NER corpus_a …")
    cnt_a = collect_ner_entities(CORPUS_A, nlp)
    print("NER corpus_b …")
    cnt_b = collect_ner_entities(CORPUS_B, nlp)

    nua, nub = len(cnt_a), len(cnt_b)
    print(
        f"Entities: A — {nua} types, {sum(cnt_a.values())} spans; "
        f"B — {nub} types, {sum(cnt_b.values())} spans."
    )

    print("RF2 descriptions …")
    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    print(f"  {len(exact_map):,} exact keys; {len(rows):,} EN rows.")
    inv = index_description_rows(rows)

    print("RF2 is-a …")
    isa_parents = load_rf2_isa_parent_map(rel_files)
    depth_memo: dict[str, int | None] = {}

    out_rows: list[dict[str, str | int]] = []
    mdepth_a = mdepth_b = mconc_a = mconc_b = 0

    def process_corpus(counter: Counter[str], label: str) -> None:
        nonlocal mdepth_a, mdepth_b, mconc_a, mconc_b
        for entity, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            cid = best_snomed_match(entity, exact_map, rows, inv)
            depth_val: int | str = ""
            if cid:
                if label == "A":
                    mconc_a += 1
                else:
                    mconc_b += 1
                d = max_depth_from_root_rf2(cid, isa_parents, memo=depth_memo)
                if d is not None:
                    depth_val = d
                    if label == "A":
                        mdepth_a += 1
                    else:
                        mdepth_b += 1
            out_rows.append(
                {
                    "entity": entity,
                    "corpus": label,
                    "frequency": freq,
                    "snomed_concept": cid or "",
                    "depth": depth_val,
                }
            )

    process_corpus(cnt_a, "A")
    process_corpus(cnt_b, "B")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["entity", "corpus", "frequency", "snomed_concept", "depth"],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {OUT_CSV.name}")
    print(
        f"Concept match: A {mconc_a}/{nua} = {mconc_a / nua if nua else 0:.3f}; "
        f"B {mconc_b}/{nub} = {mconc_b / nub if nub else 0:.3f}"
    )
    print(
        f"With depth: A {mdepth_a}/{nua} = {mdepth_a / nua if nua else 0:.3f}; "
        f"B {mdepth_b}/{nub} = {mdepth_b / nub if nub else 0:.3f}"
    )
    mean_a, na = mean_depth_for_corpus(out_rows, "A")
    mean_b, nb = mean_depth_for_corpus(out_rows, "B")
    print(f"Mean depth (matched): A = {mean_a:.2f} (n={na}); B = {mean_b:.2f} (n={nb})")

    matched_rows = [r for r in out_rows if str(r["depth"]).isdigit()]
    for lab, name in ("A", "human"), ("B", "model"):
        deep, shallow = top_depth_extremes(matched_rows, lab, 10)
        print(f"\nTop 10 deepest ({lab} / {name}):")
        for r in deep:
            print(f"  depth={r['depth']}  {r['entity'][:60]}  -> {r['snomed_concept']}")
        print(f"Top 10 shallowest ({lab} / {name}):")
        for r in shallow:
            print(f"  depth={r['depth']}  {r['entity'][:60]}  -> {r['snomed_concept']}")

    lma, lmb, la_n, lb_n = load_legacy_freq_depth_means(LEGACY_DEPTH)
    print("\n--- Mean depth: frequency table vs this NER run ---")
    if lma is None:
        print(f"No {LEGACY_DEPTH.name}; run python -m pipeline.rf2 for a baseline.")
    else:
        print(
            f"Frequency: A = {lma:.2f} (n={la_n}), B = {lmb:.2f} (n={lb_n})  [{LEGACY_DEPTH.name}]"
        )
        print(f"NER:       A = {mean_a:.2f} (n={na}), B = {mean_b:.2f} (n={nb})")


if __name__ == "__main__":
    main()
