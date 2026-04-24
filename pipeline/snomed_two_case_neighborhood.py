#!/usr/bin/env python3
"""Build a hop-limited SNOMED is-a subgraph around NER-mapped concepts for **two** case twins.

Reads ``corpus_a`` / ``corpus_b`` for two stems, runs the same NER + SNOMED match as other per-case
scripts, then expands each seed concept by a bounded walk **up** (parents) and **down** (children).
Writes GraphML (Gephi, yEd; or Neo4j via ``apoc.import.graphml``) and a static PNG (NetworkX + matplotlib).

Example::

    python -m pipeline.snomed_two_case_neighborhood \\
        case_001.txt case_005.txt --rf2-root \"$RF2\"
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .graph_viz import load_rf2_labels_for_concepts
from .ner_depth import (
    _ensure_python,
    _load_nlp,
    best_snomed_match,
    collect_ner_entities_for_document,
    index_description_rows,
    load_rf2_en_description_rows,
    load_rf2_term_to_concept,
)
from .paths import CORPUS_A, CORPUS_B, DEFAULT_RF2_ROOT, DIR_SNOMED, ensure_results
from .rf2 import (
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
)


def _invert_parents(parents: dict[str, list[str]]) -> dict[str, list[str]]:
    ch: dict[str, list[str]] = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            ch[p].append(c)
    return {k: sorted(set(v)) for k, v in ch.items()}


def _bfs_ancestors_capped(seeds: set[str], parents: dict[str, list[str]], max_hops: int, max_nodes: int) -> set[str]:
    """Breadth-first union from all seeds toward parents; stop at hop or global node budget."""
    if max_hops <= 0 or not seeds:
        return set(seeds)
    out: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(max_hops):
        if len(out) >= max_nodes:
            break
        nxt: set[str] = set()
        for n in frontier:
            for p in parents.get(n, []):
                if p not in out:
                    out.add(p)
                    nxt.add(p)
                    if len(out) >= max_nodes:
                        break
            if len(out) >= max_nodes:
                break
        frontier = nxt
        if not frontier:
            break
    return out


def _bfs_descendants_capped(
    seeds: set[str],
    children: dict[str, list[str]],
    max_hops: int,
    max_nodes: int,
    *,
    already: set[str] | None = None,
) -> set[str]:
    """Breadth-first from seeds toward children; stops at hop limit or when ``max_nodes`` nodes in ``out|already``."""
    if max_hops <= 0 or not seeds:
        return set(seeds)
    out: set[str] = set(seeds)
    frontier = set(seeds)
    seen_other = already or set()
    for _ in range(max_hops):
        if len(out | seen_other) >= max_nodes:
            break
        nxt: set[str] = set()
        for n in frontier:
            for c in children.get(n, []):
                if c in out:
                    continue
                if len(out | seen_other) >= max_nodes:
                    break
                out.add(c)
                nxt.add(c)
            if len(out | seen_other) >= max_nodes:
                break
        frontier = nxt
        if not frontier:
            break
    return out


def _mapped_concepts_for_case(
    stem: str,
    corpus_a: Path,
    corpus_b: Path,
    exact_map: dict[str, str],
    rows: list,
    inv: dict,
    parents: dict[str, list[str]],
    nlp,
) -> tuple[set[str], dict[str, set[str]]]:
    """Return concept ids and origin map cid -> {'A','B'}."""
    path_a = corpus_a / stem
    path_b = corpus_b / stem
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"Missing twin for {stem}")
    text_a = path_a.read_text(encoding="utf-8", errors="replace")
    text_b = path_b.read_text(encoding="utf-8", errors="replace")
    ca = collect_ner_entities_for_document(text_a, nlp)
    cb = collect_ner_entities_for_document(text_b, nlp)
    origins: dict[str, set[str]] = defaultdict(set)
    concepts: set[str] = set()
    for ent, _ in ca.items():
        el = ent.strip().lower()
        if not el:
            continue
        cid = best_snomed_match(el, exact_map, rows, inv)
        if cid:
            concepts.add(cid)
            origins[cid].add("A")
    for ent, _ in cb.items():
        el = ent.strip().lower()
        if not el:
            continue
        cid = best_snomed_match(el, exact_map, rows, inv)
        if cid:
            concepts.add(cid)
            origins[cid].add("B")
    return concepts, dict(origins)


def _build_digraph(nodes: set[str], parents: dict[str, list[str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for n in nodes:
        for p in parents.get(n, []):
            if p in nodes:
                G.add_edge(n, p, rel="is_a")
    return G


def _slug_two_cases(a: str, b: str) -> str:
    def one(s: str) -> str:
        m = re.search(r"(\d+)", s)
        return f"{int(m.group(1)):03d}" if m else re.sub(r"[^\w]+", "_", s)[:20]

    x, y = one(a), one(b)
    return f"{x}_{y}" if x <= y else f"{y}_{x}"


def main() -> None:
    ensure_results()
    _ensure_python()
    p = argparse.ArgumentParser(
        description="Hop-limited SNOMED is-a subgraph for two case files (A+B NER seeds)."
    )
    p.add_argument(
        "cases",
        nargs=2,
        metavar="CASE",
        help="Two case filenames, e.g. case_001.txt case_005.txt",
    )
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument("--up-hops", type=int, default=3, help="Max is-a steps toward parents (broader).")
    p.add_argument("--down-hops", type=int, default=1, help="Max is-a steps toward children (narrower); SNOMED fans out fast.")
    p.add_argument(
        "--max-nodes",
        type=int,
        default=320,
        metavar="N",
        help="Hard cap on nodes kept when expanding (ancestors share budget with descendants).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DIR_SNOMED,
        help="Directory for GraphML + PNG (default: results/snomed/).",
    )
    args = p.parse_args()

    stem_a, stem_b = args.cases[0].strip(), args.cases[1].strip()
    if not stem_a.endswith(".txt"):
        stem_a = f"case_{int(stem_a):03d}.txt" if stem_a.isdigit() else stem_a
    if not stem_b.endswith(".txt"):
        stem_b = f"case_{int(stem_b):03d}.txt" if stem_b.isdigit() else stem_b

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    parents = load_rf2_isa_parent_map(rel_files)
    children = _invert_parents(parents)
    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    inv = index_description_rows(rows)

    print("Loading en_core_sci_lg …", file=sys.stderr)
    nlp = _load_nlp()

    seeds_a, origin_a = _mapped_concepts_for_case(
        stem_a, args.corpus_a, args.corpus_b, exact_map, rows, inv, parents, nlp
    )
    seeds_b, origin_b = _mapped_concepts_for_case(
        stem_b, args.corpus_a, args.corpus_b, exact_map, rows, inv, parents, nlp
    )
    all_seeds = seeds_a | seeds_b
    if not all_seeds:
        raise SystemExit("No mapped SNOMED concepts on either case; check NER / RF2.")

    origins_merged: dict[str, set[str]] = defaultdict(set)
    case_of_seed: dict[str, set[str]] = defaultdict(set)
    for cid, sides in origin_a.items():
        origins_merged[cid] |= sides
        case_of_seed[cid].add(stem_a)
    for cid, sides in origin_b.items():
        origins_merged[cid] |= sides
        case_of_seed[cid].add(stem_b)

    cap = max(60, args.max_nodes)
    nodes = _bfs_ancestors_capped(all_seeds, parents, args.up_hops, cap)
    if args.down_hops > 0:
        extra = _bfs_descendants_capped(
            all_seeds, children, args.down_hops, cap, already=set(nodes)
        )
        nodes |= extra

    if len(nodes) >= cap - 3:
        print(
            f"Note: graph near node cap ({cap}); raise --max-nodes or hops for more context.",
            file=sys.stderr,
        )

    G = _build_digraph(nodes, parents)
    want_labels = set(G.nodes)
    labels = load_rf2_labels_for_concepts(desc_files, want_labels)

    for cid in G.nodes:
        sides = ",".join(sorted(origins_merged.get(cid, set())))
        cases = "+".join(sorted(case_of_seed.get(cid, set())))
        G.nodes[cid]["sctid"] = cid
        G.nodes[cid]["label"] = (labels.get(cid, cid) or cid)[:120]
        G.nodes[cid]["seed_sides"] = sides or ""
        G.nodes[cid]["seed_cases"] = cases or ""

    slug = _slug_two_cases(stem_a, stem_b)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = out_dir / f"snomed_two_case_neighborhood_{slug}.graphml"
    png_path = out_dir / f"snomed_two_case_neighborhood_{slug}.png"

    nx.write_graphml(G, graphml_path)

    # Layout + PNG
    pos = nx.spring_layout(G, k=0.42, iterations=28, seed=42)
    plt.figure(figsize=(14, 10), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("#fafafa")

    seed_set = all_seeds
    colors = []
    for n in G.nodes:
        if n in seed_set:
            if origins_merged.get(n) == {"A"}:
                colors.append("#0d47a1")
            elif origins_merged.get(n) == {"B"}:
                colors.append("#b71c1c")
            else:
                colors.append("#6a1b9a")
        else:
            colors.append("#9e9e9e")

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.6, arrows=True, arrowsize=8, edge_color="#555")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=120, alpha=0.9)
    short_labels = {}
    for n in G.nodes:
        lab = G.nodes[n].get("label", n)
        short_labels[n] = (lab[:38] + "…") if len(str(lab)) > 40 else str(lab)
    nx.draw_networkx_labels(G, pos, labels=short_labels, font_size=5, font_color="#111")

    plt.title(
        f"SNOMED is-a neighbourhood (up {args.up_hops} / down {args.down_hops} hops)\n"
        f"{stem_a} + {stem_b} — blue=A seed, red=B seed, purple=both, grey=expanded",
        fontsize=11,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, facecolor="white")
    plt.close()

    print(f"Wrote {graphml_path}", file=sys.stderr)
    print(f"Wrote {png_path}", file=sys.stderr)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}  Seeds: {len(all_seeds)}", file=sys.stderr)


if __name__ == "__main__":
    main()
