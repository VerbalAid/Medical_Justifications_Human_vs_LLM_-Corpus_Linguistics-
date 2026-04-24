#!/usr/bin/env python3
# One network picture: part of the SNOMED is-a tree around the deepest NER hits.

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .paths import SNOMED_DEPTH_NER, SNOMED_SUBGRAPH, ensure_results, explain_missing_ner_csv
from .rf2 import (
    DEFAULT_RF2_ROOT,
    RF2_FSN_TYPE_ID,
    RF2_ROOT_CONCEPT_ID,
    RF2_SYNONYM_TYPE_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
)

SCT_ROOT = RF2_ROOT_CONCEPT_ID
DEFAULT_NER_CSV = SNOMED_DEPTH_NER
OUT_PNG = SNOMED_SUBGRAPH


def _parse_depth(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s or not s.isdigit():
        return None
    return int(s)


def read_top_matched_rows(csv_path: Path, corpus: str, k: int, min_depth: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("corpus") != corpus:
                continue
            if not (row.get("snomed_concept") or "").strip():
                continue
            d = _parse_depth(row.get("depth", ""))
            if d is None or d < min_depth:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (-int(r["frequency"]), r.get("entity", "")))
    return rows[:k]


def _csv_field_limit() -> None:
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def load_rf2_labels_for_concepts(description_paths: list[Path], want: set[str]) -> dict[str, str]:
    if not want:
        return {}
    _csv_field_limit()
    best: dict[str, tuple[str, int]] = {}
    rank_fsn, rank_syn = 0, 1
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
                if row[ac] != "1":
                    continue
                cid = row[ci]
                if cid not in want:
                    continue
                if not row[lc].strip().lower().startswith("en"):
                    continue
                typ = row[ti]
                if typ not in (RF2_FSN_TYPE_ID, RF2_SYNONYM_TYPE_ID):
                    continue
                key = row[te].strip().lower()
                if not key:
                    continue
                rank = rank_fsn if typ == RF2_FSN_TYPE_ID else rank_syn
                prev = best.get(cid)
                if prev is None or rank < prev[1]:
                    best[cid] = (key, rank)
    return {cid: t for cid, (t, _) in best.items()}


def collect_ancestor_nodes(seeds: set[str], child_to_parents: dict[str, list[str]]) -> set[str]:
    """Full upward closure to the root (legacy behaviour; can be huge)."""
    out: set[str] = set()
    stack = list(seeds)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(child_to_parents.get(n, []))
    return out


def collect_ancestor_nodes_hop_limited(
    seeds: set[str],
    child_to_parents: dict[str, list[str]],
    max_hops: int,
) -> set[str]:
    """Ancestors within ``max_hops`` parent edges of any seed (readable subgraphs)."""
    if max_hops < 0:
        return collect_ancestor_nodes(seeds, child_to_parents)
    dist: dict[str, int] = {}
    dq: deque[tuple[str, int]] = deque()
    for s in seeds:
        dist[s] = 0
        dq.append((s, 0))
    out: set[str] = set()
    while dq:
        n, d = dq.popleft()
        if dist.get(n, 10**9) != d:
            continue
        out.add(n)
        if d >= max_hops:
            continue
        for p in child_to_parents.get(n, []):
            nd = d + 1
            if nd > max_hops:
                continue
            if p not in dist or nd < dist[p]:
                dist[p] = nd
                dq.append((p, nd))
    return out


def build_subgraph(nodes: set[str], child_to_parents: dict[str, list[str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for n in nodes:
        for p in child_to_parents.get(n, []):
            if p in nodes:
                G.add_edge(n, p)
    return G


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(
        description=(
            "SNOMED is-a subgraph PNG from NER depth CSV. "
            "Use --compact or --ancestor-hops for a small interpretable slice; "
            "the default is still full ancestor closure (dense)."
        )
    )
    parser.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    parser.add_argument("--ner-csv", type=Path, default=DEFAULT_NER_CSV)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-depth", type=int, default=6)
    parser.add_argument(
        "--ancestor-hops",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max parent hops upward from each seed (omit for full closure to root). "
            "Try 8–14 for a readable figure."
        ),
    )
    parser.add_argument(
        "--label-seeds-only",
        action="store_true",
        help="Draw text labels only on seed entities; grey ancestors stay unlabelled.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Shorthand: --top 5 --ancestor-hops 12 --label-seeds-only (override with explicit flags).",
    )
    parser.add_argument("--out", type=Path, default=OUT_PNG)
    args = parser.parse_args()

    if args.compact:
        if args.top == 20:
            args.top = 5
        if args.ancestor_hops is None:
            args.ancestor_hops = 12
        args.label_seeds_only = True

    if not args.ner_csv.is_file():
        raise SystemExit(explain_missing_ner_csv(args.ner_csv))

    top_a = read_top_matched_rows(args.ner_csv, "A", args.top, args.min_depth)
    top_b = read_top_matched_rows(args.ner_csv, "B", args.top, args.min_depth)
    if not top_a and not top_b:
        raise SystemExit(
            f"No rows in {args.ner_csv} with concept and depth >= {args.min_depth} "
            "(lower --min-depth or regenerate NER CSV)."
        )

    print(
        f"Seeds: A {len(top_a)}, B {len(top_b)} (depth >= {args.min_depth}, top {args.top} by freq).",
        file=sys.stderr,
    )

    seeds_a = {r["snomed_concept"].strip() for r in top_a}
    seeds_b = {r["snomed_concept"].strip() for r in top_b}
    seeds_all = seeds_a | seeds_b

    entity_label_a: dict[str, str] = {}
    for r in top_a:
        cid = r["snomed_concept"].strip()
        entity_label_a.setdefault(cid, r.get("entity", cid))
    entity_label_b: dict[str, str] = {}
    for r in top_b:
        cid = r["snomed_concept"].strip()
        entity_label_b.setdefault(cid, r.get("entity", cid))

    freq_a: dict[str, int] = {}
    for r in top_a:
        cid = r["snomed_concept"].strip()
        freq_a[cid] = max(freq_a.get(cid, 0), int(r["frequency"]))
    freq_b: dict[str, int] = {}
    for r in top_b:
        cid = r["snomed_concept"].strip()
        freq_b[cid] = max(freq_b.get(cid, 0), int(r["frequency"]))

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))
    if not rf2_root.is_dir():
        raise SystemExit(f"RF2 root must be a directory: {rf2_root}")

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need Description and Relationship under --rf2-root.")

    print("Loading RF2 is-a …", file=sys.stderr)
    child_to_parents = load_rf2_isa_parent_map(rel_files)

    if args.ancestor_hops is not None:
        nodes = collect_ancestor_nodes_hop_limited(seeds_all, child_to_parents, args.ancestor_hops)
        print(
            f"Ancestor hop cap: {args.ancestor_hops} (subset of full closure; root may be absent).",
            file=sys.stderr,
        )
    else:
        nodes = collect_ancestor_nodes(seeds_all, child_to_parents)
    if SCT_ROOT not in nodes and any(seeds_all):
        print("Warning: SNOMED root not in ancestor closure; graph may be disconnected.", file=sys.stderr)

    G = build_subgraph(nodes, child_to_parents)
    if G.number_of_nodes() == 0:
        raise SystemExit("Empty subgraph.")

    print(f"Labels for {len(nodes):,} concepts …", file=sys.stderr)
    cid_to_term = load_rf2_labels_for_concepts(desc_files, nodes)

    labels: dict[str, str] = {}
    for n in G.nodes():
        if n in seeds_a and n in seeds_b:
            labels[n] = entity_label_a.get(n) or entity_label_b.get(n) or cid_to_term.get(n, n)
        elif n in seeds_a:
            labels[n] = entity_label_a.get(n, cid_to_term.get(n, n))
        elif n in seeds_b:
            labels[n] = entity_label_b.get(n, cid_to_term.get(n, n))
        else:
            labels[n] = cid_to_term.get(n, n)

    def short_label(s: str, max_len: int = 32) -> str:
        s = s.replace("\n", " ")
        return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"

    labels = {k: short_label(v) for k, v in labels.items()}

    if args.label_seeds_only:
        labels_draw = {n: labels[n] for n in seeds_all if n in labels}
    else:
        labels_draw = labels

    colors: list[str] = []
    sizes: list[float] = []
    for n in G.nodes():
        in_a, in_b = n in seeds_a, n in seeds_b
        if in_a and in_b:
            colors.append("#7B1FA2")
        elif in_a:
            colors.append("#1565C0")
        elif in_b:
            colors.append("#C62828")
        else:
            colors.append("#B0BEC5")
        f = max(freq_a.get(n, 0), freq_b.get(n, 0))
        sizes.append(400.0 + 35.0 * math.sqrt(float(f)) if f > 0 else 220.0)

    n_nodes = G.number_of_nodes()
    fig_w, fig_h = (11, 8) if n_nodes < 80 else (14, 10)
    plt.figure(figsize=(fig_w, fig_h))
    k_spread = 0.9 if n_nodes < 40 else (1.2 if n_nodes < 120 else 1.8)
    pos = nx.spring_layout(G, k=k_spread, iterations=80 if n_nodes < 150 else 50, seed=42)
    ewidth = 1.0 if n_nodes < 80 else 0.5
    ealpha = 0.75 if n_nodes < 120 else 0.35
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#90A4AE",
        arrows=True,
        arrowsize=10 if n_nodes < 100 else 8,
        width=ewidth,
        alpha=ealpha,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.92)
    lbl_fs = 9 if args.label_seeds_only else (8 if n_nodes < 100 else 6)
    nx.draw_networkx_labels(G, pos, labels_draw, font_size=lbl_fs, font_family="sans-serif")
    hop_note = f" (ancestors within {args.ancestor_hops} hops)" if args.ancestor_hops is not None else " (ancestors to root)"
    lbl_note = "; labels on seeds only" if args.label_seeds_only else ""
    plt.title(
        "SNOMED CT is-a subgraph: top NER entities"
        + hop_note
        + "\nBlue = A only, red = B only, purple = both, grey = ancestor only"
        + lbl_note
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges).")


if __name__ == "__main__":
    main()
