#!/usr/bin/env python3
"""Visual graph of how one case's **explanation text** (default: model / corpus B) hangs on SNOMED.

Nodes
-----
* **Sentence** — spaCy sentence spans from the explanation prose.
* **Entity** — distinct NER surfaces from that text.
* **Concept** — SNOMED IDs mapped from entities, plus a small upward ``is_a`` hull (parents).

Edges: ``Sentence --mentions--> Entity``, ``Entity --maps_to--> Concept``, ``Concept --is_a--> Concept``.

This is a **read-only structural picture** of how the wording sits on the ontology for one twin file.
For an LLM-over-graph **judge** pass (JSON verdict), run ``python -m pipeline.model_explanation_graph_rag_judge``.

Optional ``--compare-a`` draws **two** panels (human vs model) for the same ``case_NNN.txt``.

Example::

    python -m pipeline.model_explanation_snomed_graph case_005.txt --compare-a --rf2-root \"$RF2\"
"""

from __future__ import annotations

import argparse
import re
import sys
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
from .rf2 import discover_rf2_description_and_relationship_files, format_rf2_root_missing_error, load_rf2_isa_parent_map


def _text_has_entity(text: str, entity: str) -> bool:
    e = (entity or "").strip()
    if not e:
        return False
    tl = text.lower()
    el = e.lower()
    if " " in el or "-" in el or len(el) < 2:
        return el in tl
    return re.search(r"(?<![a-z0-9])" + re.escape(el) + r"(?![a-z0-9])", tl) is not None


def _slug(stem: str) -> str:
    m = re.search(r"(\d+)", stem)
    return f"{int(m.group(1)):03d}" if m else re.sub(r"[^\w]+", "_", stem)[:24]


def _expand_concepts(concept_ids: set[str], parents: dict[str, list[str]], max_hops: int, cap: int) -> set[str]:
    out = set(concept_ids)
    frontier = set(concept_ids)
    for _ in range(max_hops):
        if len(out) >= cap:
            break
        nxt: set[str] = set()
        for c in frontier:
            for p in parents.get(c, []):
                if p not in out:
                    out.add(p)
                    nxt.add(p)
                    if len(out) >= cap:
                        break
            if len(out) >= cap:
                break
        frontier = nxt
        if not frontier:
            break
    return out


def build_explanation_graph(
    text: str,
    nlp,
    *,
    prefix: str,
    exact_map: dict[str, str],
    rows: list,
    inv: dict,
    parents: dict[str, list[str]],
    labels: dict[str, str],
    parent_hops: int,
    max_concepts: int,
) -> nx.DiGraph:
    G = nx.DiGraph()
    doc = nlp(text or "")
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents and (text or "").strip():
        sents = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]

    counts = collect_ner_entities_for_document(text, nlp)
    ent_to_cid: dict[str, str] = {}
    for ent in counts:
        el = ent.strip().lower()
        if not el:
            continue
        cid = best_snomed_match(el, exact_map, rows, inv)
        if cid:
            ent_to_cid[el] = cid

    concept_ids = set(ent_to_cid.values())
    concepts = _expand_concepts(concept_ids, parents, parent_hops, max_concepts)

    for i, s in enumerate(sents):
        sid = f"{prefix}S{i}"
        G.add_node(sid, kind="sentence", label=(s[:200] + "…") if len(s) > 200 else s)

    for el, cid in ent_to_cid.items():
        eid = f"{prefix}E:{el}"
        cidn = f"{prefix}C:{cid}"
        G.add_node(eid, kind="entity", label=el)
        G.add_node(cidn, kind="concept", sctid=cid, label=(labels.get(cid, cid) or cid)[:100])
        G.add_edge(eid, cidn, rel="maps_to")
        for i, sent in enumerate(sents):
            if _text_has_entity(sent, el):
                G.add_edge(f"{prefix}S{i}", eid, rel="mentions")

    for cid in concepts:
        cidn = f"{prefix}C:{cid}"
        if cidn not in G:
            G.add_node(cidn, kind="concept", sctid=cid, label=(labels.get(cid, cid) or cid)[:100])
    for cid in concepts:
        cidn = f"{prefix}C:{cid}"
        for p in parents.get(cid, []):
            if p not in concepts:
                continue
            pid = f"{prefix}C:{p}"
            G.add_edge(cidn, pid, rel="is_a")

    return G


def _draw_panel(ax, G: nx.DiGraph, title: str) -> None:
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(empty graph)", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.axis("off")
        return

    pos = nx.spring_layout(G, k=0.55, iterations=35, seed=7)
    colors = []
    sizes = []
    for n in G.nodes:
        k = G.nodes[n].get("kind", "")
        if k == "sentence":
            colors.append("#fff3bf")
            sizes.append(380)
        elif k == "entity":
            colors.append("#a5d6a7")
            sizes.append(260)
        else:
            colors.append("#90caf9")
            sizes.append(220)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=0.7, arrows=True, arrowsize=10, edge_color="#555")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.92)
    short = {}
    for n in G.nodes:
        lab = G.nodes[n].get("label", n)
        short[n] = (str(lab)[:32] + "…") if len(str(lab)) > 34 else str(lab)
    nx.draw_networkx_labels(G, pos, labels=short, ax=ax, font_size=5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def main() -> None:
    ensure_results()
    _ensure_python()
    p = argparse.ArgumentParser(
        description="Graph: sentences → NER entities → SNOMED (+ short is-a hull) for model (B) text."
    )
    p.add_argument("case", help="Case stem, e.g. case_005.txt or 005")
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument(
        "--compare-a",
        action="store_true",
        help="Also graph corpus A for the same case (two-panel figure).",
    )
    p.add_argument("--parent-hull-hops", type=int, default=2, help="Parents added per mapped concept.")
    p.add_argument("--max-concepts", type=int, default=90, help="Cap total concept nodes (expansion budget).")
    p.add_argument("--out-dir", type=Path, default=DIR_SNOMED)
    args = p.parse_args()

    stem = args.case.strip()
    if not stem.endswith(".txt"):
        stem = f"case_{int(stem):03d}.txt" if stem.isdigit() else stem

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))
    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    parents = load_rf2_isa_parent_map(rel_files)
    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    inv = index_description_rows(rows)

    path_b = args.corpus_b / stem
    if not path_b.is_file():
        raise SystemExit(f"Missing {path_b}")
    text_b = path_b.read_text(encoding="utf-8", errors="replace")
    text_a = ""
    if args.compare_a:
        path_a = args.corpus_a / stem
        if path_a.is_file():
            text_a = path_a.read_text(encoding="utf-8", errors="replace")

    print("Loading en_core_sci_lg …", file=sys.stderr)
    nlp = _load_nlp()

    # Preload labels for any concept we might touch (mapped + small hull)
    want: set[str] = set()

    def _grow_want_from_text(t: str, bag: set[str]) -> None:
        tmp = collect_ner_entities_for_document(t, nlp)
        for ent in tmp:
            cid = best_snomed_match(ent.strip().lower(), exact_map, rows, inv)
            if cid:
                bag.add(cid)
        for _ in range(args.parent_hull_hops + 2):
            nxt = set()
            for c in list(bag):
                for pa in parents.get(c, []):
                    nxt.add(pa)
            bag |= nxt
            if len(bag) > 8000:
                break

    _grow_want_from_text(text_b, want)
    if text_a:
        _grow_want_from_text(text_a, want)
    labels = load_rf2_labels_for_concepts(desc_files, want)

    Gb = build_explanation_graph(
        text_b,
        nlp,
        prefix="B_",
        exact_map=exact_map,
        rows=rows,
        inv=inv,
        parents=parents,
        labels=labels,
        parent_hops=args.parent_hull_hops,
        max_concepts=args.max_concepts,
    )

    slug = _slug(stem)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    graphml_b = out_dir / f"model_explanation_graph_{slug}_B.graphml"
    nx.write_graphml(Gb, graphml_b)

    if args.compare_a:
        path_a = args.corpus_a / stem
        if not path_a.is_file() or not text_a:
            raise SystemExit(f"--compare-a but missing or empty {path_a}")
        Ga = build_explanation_graph(
            text_a,
            nlp,
            prefix="A_",
            exact_map=exact_map,
            rows=rows,
            inv=inv,
            parents=parents,
            labels=labels,
            parent_hops=args.parent_hull_hops,
            max_concepts=args.max_concepts,
        )
        nx.write_graphml(Ga, out_dir / f"model_explanation_graph_{slug}_A.graphml")
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor="white")
        _draw_panel(axes[0], Ga, f"Human (A) — {stem}")
        _draw_panel(axes[1], Gb, f"Model (B) — {stem}")
        fig.suptitle(
            "Explanation as a graph: sentences → entities → SNOMED (+ short is-a hull)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        png_path = out_dir / f"model_explanation_graph_{slug}_AB.png"
    else:
        fig, ax = plt.subplots(figsize=(12.5, 9), facecolor="white")
        _draw_panel(ax, Gb, f"Model (B) — {stem}")
        fig.suptitle(
            "Model explanation as a graph: sentences → entities → SNOMED (+ short is-a hull)",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        png_path = out_dir / f"model_explanation_graph_{slug}_B.png"

    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Wrote {graphml_b}", file=sys.stderr)
    print(f"Wrote {png_path}", file=sys.stderr)
    print(
        "Legend: yellow = sentence, green = NER entity, blue = SNOMED concept; "
        "arrows: mentions / maps_to / is_a.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
