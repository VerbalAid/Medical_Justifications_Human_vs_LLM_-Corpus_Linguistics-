#!/usr/bin/env python3
# Run ner_depth then every step that needs its CSV, in the right order.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .paths import DEFAULT_RF2_ROOT, SNOMED_DEPTH_NER, explain_missing_ner_csv


def _run(module: str, argv: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run pipeline.ner_depth, then graph_viz, pair_depths, branch_pairs, "
            "snomed_taxonomy_compare (same order you need after rf2)."
        )
    )
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument(
        "--skip-ner",
        action="store_true",
        help="Skip ner_depth (use if snomed_depth_from_ner_entities.csv already exists).",
    )
    p.add_argument(
        "--graph-compact",
        action="store_true",
        help="Pass --compact to graph_viz (smaller hop-limited subgraph, seed labels only).",
    )
    args = p.parse_args()
    rf2 = str(args.rf2_root.resolve())
    rf2_args = ["--rf2-root", rf2]

    if sys.version_info >= (3, 14):
        raise SystemExit(
            "spaCy does not support Python 3.14 yet. Recreate the venv with 3.12, then:\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python -m pipeline.run_snomed_ner_chain --rf2-root <RF2>\n"
        )

    if args.skip_ner and not SNOMED_DEPTH_NER.is_file():
        raise SystemExit(explain_missing_ner_csv())

    if not args.skip_ner:
        _run("pipeline.ner_depth", rf2_args)

    gv_args = list(rf2_args)
    if args.graph_compact:
        gv_args.append("--compact")
    _run("pipeline.graph_viz", gv_args)
    _run("pipeline.pair_depths", [])
    _run("pipeline.branch_pairs", rf2_args + ["--dedupe-concept-pairs"])
    _run("pipeline.snomed_taxonomy_compare", rf2_args)
    print("\nSNOMED NER chain finished.", flush=True)


if __name__ == "__main__":
    main()
