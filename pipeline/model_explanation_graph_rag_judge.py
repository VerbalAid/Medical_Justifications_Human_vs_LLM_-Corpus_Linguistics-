#!/usr/bin/env python3
"""Graph-style “RAG” over the explanation structure + **LLM JSON judge** (Ollama).

Builds the same sentence → entity → SNOMED graph as ``model_explanation_snomed_graph`` (for B, and
optionally A), serialises it to a compact ``node_link`` JSON bundle, and asks Mistral to score how
well the **ontology subgraph supports the prose** (not clinical truth against an external gold).

Output: one JSON file under ``results/llm_align/`` with the serialised graphs, model verdict, and
metadata. Requires a running Ollama instance (same as ``llm_pair_align``).

Example::

    python -m pipeline.model_explanation_graph_rag_judge case_005.txt --include-human --rf2-root \"$RF2\"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import ollama
from networkx.readwrite import json_graph

from .graph_viz import load_rf2_labels_for_concepts
from .llm_pair_align import parse_llm_json
from .model_explanation_snomed_graph import build_explanation_graph
from .ner_depth import (
    _ensure_python,
    _load_nlp,
    best_snomed_match,
    collect_ner_entities_for_document,
    index_description_rows,
    load_rf2_en_description_rows,
    load_rf2_term_to_concept,
)
from .paths import CORPUS_A, CORPUS_B, DEFAULT_RF2_ROOT, DIR_LLM_ALIGN, ensure_results
from .rf2 import discover_rf2_description_and_relationship_files, format_rf2_root_missing_error, load_rf2_isa_parent_map

MODEL = "mistral"
SLEEP_SEC = 0.45

JUDGE_SYSTEM = (
    "You are a corpus-linguistics and terminology reviewer. You receive a **directed graph** built "
    "automatically from clinical explanation prose: sentence nodes, NER entity nodes, SNOMED CT "
    "concept nodes, and edges mentions / maps_to / is_a. The matcher is imperfect; treat the graph "
    "as evidence of how the text was *packaged onto* the ontology, not as ground-truth clinical "
    "classification. Reply with **one JSON object only** (no markdown fences), British English in "
    "string values."
)


def _slug(stem: str) -> str:
    m = re.search(r"(\d+)", stem)
    return f"{int(m.group(1)):03d}" if m else re.sub(r"[^\w]+", "_", stem)[:24]


def _grow_want(
    t: str,
    nlp,
    exact_map,
    rows,
    inv,
    parents: dict[str, list[str]],
    parent_hull_hops: int,
    cap: int,
) -> set[str]:
    bag: set[str] = set()
    tmp = collect_ner_entities_for_document(t, nlp)
    for ent in tmp:
        cid = best_snomed_match(ent.strip().lower(), exact_map, rows, inv)
        if cid:
            bag.add(cid)
    for _ in range(parent_hull_hops + 2):
        nxt = set()
        for c in list(bag):
            for pa in parents.get(c, []):
                nxt.add(pa)
        bag |= nxt
        if len(bag) >= cap:
            break
    return bag


def _grow_want_union(
    texts: list[str],
    nlp,
    exact_map,
    rows,
    inv,
    parents: dict[str, list[str]],
    parent_hull_hops: int,
) -> set[str]:
    out: set[str] = set()
    for t in texts:
        if not t.strip():
            continue
        out |= _grow_want(t, nlp, exact_map, rows, inv, parents, parent_hull_hops, 600)
    return out


def _trim_node_link(data: dict[str, Any], max_nodes: int) -> dict[str, Any]:
    nodes = list(data.get("nodes", []))
    links = list(data.get("links", []))
    if len(nodes) <= max_nodes:
        return data

    by_kind: dict[str, list[dict[str, Any]]] = {"concept": [], "entity": [], "sentence": [], "other": []}
    for n in nodes:
        k = (n.get("kind") or "").strip() or "other"
        by_kind.setdefault(k if k in by_kind else "other", []).append(n)

    kept: list[dict[str, Any]] = []
    kept.extend(by_kind.get("concept", []))
    kept.extend(by_kind.get("entity", []))
    budget = max_nodes - len(kept)
    sents = by_kind.get("sentence", [])
    if budget > 0 and sents:
        def deg(n: dict[str, Any]) -> int:
            nid = n.get("id")
            return sum(1 for lk in links if lk.get("source") == nid or lk.get("target") == nid)

        sents.sort(key=lambda n: -deg(n))
        kept.extend(sents[:budget])

    keep_ids = {n.get("id") for n in kept if n.get("id") is not None}
    new_links = [
        lk
        for lk in links
        if lk.get("source") in keep_ids and lk.get("target") in keep_ids
    ]
    return {"directed": data.get("directed", True), "multigraph": data.get("multigraph", False), "graph": data.get("graph", {}), "nodes": kept, "links": new_links}


def _graph_payload(G: nx.DiGraph, *, max_nodes: int) -> dict[str, Any]:
    raw = json_graph.node_link_data(G)
    return _trim_node_link(raw, max_nodes)


def _call_judge(user: str, *, model: str) -> dict[str, Any]:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.2, "num_predict": 900},
    )
    msg = getattr(resp, "message", None)
    content = (getattr(msg, "content", None) or "").strip()
    if not content:
        raise RuntimeError("empty model response")
    return parse_llm_json(content)


def main() -> None:
    ensure_results()
    _ensure_python()
    p = argparse.ArgumentParser(
        description="Serialise explanation→SNOMED graph(s) and run an Ollama JSON judge."
    )
    p.add_argument("case", help="case_005.txt or 005")
    p.add_argument("--corpus-a", type=Path, default=CORPUS_A)
    p.add_argument("--corpus-b", type=Path, default=CORPUS_B)
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument(
        "--include-human",
        action="store_true",
        help="Include corpus A graph + excerpt in the judge bundle (comparative).",
    )
    p.add_argument("--parent-hull-hops", type=int, default=2)
    p.add_argument("--max-concepts", type=int, default=90)
    p.add_argument("--max-graph-nodes", type=int, default=130, help="Cap nodes sent in JSON to the LLM.")
    p.add_argument("--prose-chars", type=int, default=1800, metavar="N", help="Max chars of raw prose per side in prompt.")
    p.add_argument("--model", type=str, default=MODEL)
    p.add_argument("--sleep", type=float, default=SLEEP_SEC)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build graph bundle and write JSON without calling Ollama.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: results/llm_align/model_explanation_graph_judge_<slug>.json",
    )
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
    if args.include_human:
        path_a = args.corpus_a / stem
        if path_a.is_file():
            text_a = path_a.read_text(encoding="utf-8", errors="replace")

    print("Loading en_core_sci_lg …", file=sys.stderr)
    nlp = _load_nlp()

    texts_for_want = [text_b] + ([text_a] if text_a else [])
    want = _grow_want_union(texts_for_want, nlp, exact_map, rows, inv, parents, args.parent_hull_hops)
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
    Ga: nx.DiGraph | None = None
    if args.include_human and text_a:
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

    bundle: dict[str, Any] = {
        "case_file": stem,
        "model_graph_stats": {
            "nodes": Gb.number_of_nodes(),
            "edges": Gb.number_of_edges(),
        },
        "model_graph": _graph_payload(Gb, max_nodes=args.max_graph_nodes),
    }
    if Ga is not None:
        bundle["human_graph_stats"] = {"nodes": Ga.number_of_nodes(), "edges": Ga.number_of_edges()}
        bundle["human_graph"] = _graph_payload(Ga, max_nodes=args.max_graph_nodes)

    excerpt_b = text_b.strip()[: args.prose_chars]
    excerpt_a = text_a.strip()[: args.prose_chars] if text_a else ""

    user_obj = {
        "task": (
            "Rate how coherently the extracted graph represents the explanation: do sentences, "
            "entities, and SNOMED concepts line up as a plausible packaging? Flag suspicious "
            "entity→concept mappings or disconnected subgraph themes. If both human and model "
            "graphs are present, briefly contrast structural packaging (not who is medically correct)."
        ),
        "case_file": stem,
        "corpus_B_excerpt": excerpt_b,
        "corpus_A_excerpt": excerpt_a if args.include_human else None,
        "graph_bundle": bundle,
    }
    user = json.dumps(user_obj, ensure_ascii=False, indent=2)
    user += """

Return exactly one JSON object with keys:
  "coherence_score": integer 1-5 (graph internal consistency: mentions/maps_to/is_a hang together),
  "ontology_fit_score": integer 1-5 (how well SNOMED choices read as support for the prose; allow low if matcher noise),
  "narrative_to_graph_alignment": one sentence,
  "suspicious_mappings": array of {"entity_hint": string, "concept_label_or_id": string, "issue": string} (empty array if none),
  "disconnected_subthemes": array of short strings (empty if none),
  "human_vs_model_packaging": string or null (null unless both sides were provided),
  "verdict": one of "supported", "mixed", "weak",
  "rationale": two to four sentences, British English
"""

    slug = _slug(stem)
    out_path = args.out_json or (DIR_LLM_ALIGN / f"model_explanation_graph_judge_{slug}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "case_file": stem,
        "model": args.model,
        "include_human": bool(args.include_human),
        "input": user_obj,
    }

    if args.dry_run:
        record["judge"] = {"skipped": "dry_run"}
    else:
        try:
            verdict = _call_judge(user, model=args.model)
            record["judge"] = verdict
        except Exception as e:  # noqa: BLE001
            record["judge"] = {"error": str(e), "verdict": "error"}

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    if not args.dry_run and args.sleep > 0:
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
