#!/usr/bin/env python3
# Optional: send NER branch-pair seeds + same-case text to Ollama; write reviewed strings
# then map them to SNOMED and print is-a depths (separate from the big branch_pairs grid).

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import ollama

from .ner_depth import (
    best_snomed_match,
    index_description_rows,
    load_rf2_en_description_rows,
    load_rf2_term_to_concept,
)
from .paths import (
    CORPUS_A,
    CORPUS_B,
    DEFAULT_RF2_ROOT,
    LLM_ALIGNED_DEPTHS_CSV,
    LLM_ALIGNED_DEPTHS_PER_CASE,
    LLM_ALIGNED_DEPTHS_PER_CASE_HIERARCHICAL,
    LLM_PAIR_REVIEW_CSV,
    LLM_PAIR_REVIEW_PER_CASE,
    LLM_PAIR_REVIEW_PER_CASE_HIERARCHICAL,
    SNOMED_BRANCH_PAIRS,
    ensure_results,
)
from .rf2 import (
    RF2_ROOT_CONCEPT_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
    max_depth_from_root_rf2,
)
from .corpus_context import context_first_file_containing, context_from_case_file
from .snomed_taxonomy_compare import pick_divergent_pairs_per_category

MODEL = "mistral"
SLEEP_SEC = 0.45

SYSTEM = (
    "You are a corpus linguistics assistant. You compare two short clinical excerpts from the "
    "same case (human key vs model explanation) plus two entity strings. The strings may be "
    "**different surface forms** (e.g. a broad clinical label on one side and a narrower one on the "
    "other) when the pipeline already aligned them on an SNOMED is-a ladder. Your job is to judge "
    "whether comparing SNOMED is-a depth is still a fair read of **relative specificity** in that "
    "case (same clinical thread vs incompatible angles). If the spans are not the same clinical "
    "thing in this vignette, say INCOMPARABLE. Reply with JSON only, no markdown fences, no commentary."
)

# Per-case hierarchical seeds: user wants depth on mapped concepts, not rewritten "harmonised" strings.
SYSTEM_HIERARCHICAL = (
    "You are a corpus linguistics assistant. The pipeline paired two **different** NER strings from "
    "the same case twin whose SNOMED concepts are in a strict is-a ancestor/descendant relation. "
    "The study cares whether comparing **RF2 SNOMED depth** between those **already-mapped** concepts "
    "is meaningful for this vignette. Do **not** suggest alternative surface strings to make depth "
    "comparison easier---keep the pipeline entities unless a link is clearly wrong for the span. "
    "Reply with JSON only, no markdown fences, no commentary.\n\n"
    "KEEP: both mentions sit in the **same clinical thread** in context; comparing depth of concept A "
    "vs concept B is a fair taxonomic specificity read **as-is**.\n"
    "INCOMPARABLE: **unrelated** clinical angles, misleading to compare depth, or the pair does not "
    "reflect the same clinical focus in this vignette.\n"
    "SUBSTITUTE: **rare**. Only if an automatic SNOMED link is **clearly wrong for the text span** "
    "(wrong entity for that mention). Never use SUBSTITUTE to propose harmonised synonyms or "
    "paraphrases for depth; if wording is awkward but the thread matches, answer KEEP. If the thread "
    "does not match, answer INCOMPARABLE."
)

USER_TEMPLATE = """Same case file: {case_stem}

--- Human excerpt (corpus A) ---
{excerpt_a}

--- Model excerpt (corpus B) ---
{excerpt_b}

--- Candidate pair (often from automatic NER; may be same token in both sides or two different strings) ---
  entity_a (human side): "{ea}"
  entity_b (model side): "{eb}"

Task:
1) For this vignette, do the two strings support comparing SNOMED depth as a **specificity ladder** (broader vs narrower on the same clinical topic, or the same referent), rather than two unrelated angles?
2) If yes, is comparing SNOMED depth a fair read of **different specificity on the same taxonomic branch**? If the strings already match that goal, keep them.
3) If the clinical thread matches but the surface strings are wrong for a depth read-off, propose SHORT replacements (one noun phrase each, under six words) that harmonise the comparison.
4) If they are incompatible referents or different clinical angles, answer INCOMPARABLE.

Return exactly one JSON object with keys:
  "decision": one of "KEEP", "SUBSTITUTE", "INCOMPARABLE"
  "entity_a": string or null (null means keep "{ea}")
  "entity_b": string or null (null means keep "{eb}")
  "rationale": one sentence, British English
"""

USER_TEMPLATE_HIERARCHICAL = """Same case file: {case_stem}

--- Human excerpt (corpus A) ---
{excerpt_a}

--- Model excerpt (corpus B) ---
{excerpt_b}

--- Candidate pair (different strings; pipeline: strict SNOMED is-a ancestor/descendant) ---
  entity_a (human side): "{ea}"
  entity_b (model side): "{eb}"

Task:
1) In this vignette, do both mentions belong to the **same clinical thread** so that comparing SNOMED **depth** between the **mapped** concepts is a sensible broader-vs-narrower read?
2) If **yes**, answer **KEEP** and set **entity_a** and **entity_b** to **null** (keep originals for depth; do not rewrite strings).
3) If **no** (unrelated or misleading), answer **INCOMPARABLE** and set both entity fields to **null**.
4) **SUBSTITUTE** only if a link is **clearly wrong for the span**; then supply minimal corrected strings. If the only issue is imperfect wording but the thread matches, **KEEP** with nulls. If unrelated, **INCOMPARABLE** with nulls---not SUBSTITUTE.

Return exactly one JSON object with keys:
  "decision": one of "KEEP", "SUBSTITUTE", "INCOMPARABLE"
  "entity_a": string or null (null means keep "{ea}")
  "entity_b": string or null (null means keep "{eb}")
  "rationale": one sentence, British English
"""


def text_has_entity(text: str, entity: str) -> bool:
    e = (entity or "").strip()
    if not e:
        return False
    tl = text.lower()
    el = e.lower()
    if " " in el or "-" in el or len(el) < 2:
        return el in tl
    return (
        re.search(r"(?<![a-z0-9])" + re.escape(el) + r"(?![a-z0-9])", tl) is not None
    )


def stems_for_entity(folder: Path, entity: str) -> set[str]:
    out: set[str] = set()
    for p in sorted(folder.glob("case_*.txt")):
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text_has_entity(t, entity):
            out.add(p.name)
    return out


def pick_case_stem(entity_a: str, entity_b: str) -> str | None:
    sa = stems_for_entity(CORPUS_A, entity_a)
    sb = stems_for_entity(CORPUS_B, entity_b)
    inter = sorted(sa & sb)
    if inter:
        return inter[0]
    return None


def first_excerpt_any(folder: Path, entity: str, max_chars: int = 900) -> str:
    for p in sorted(folder.glob("case_*.txt")):
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text_has_entity(t, entity):
            t = t.strip()
            return t[:max_chars] + ("…" if len(t) > max_chars else "")
    return ""


def excerpt_for_case(stem: str, folder: Path, max_chars: int = 1200) -> str:
    if "|" in stem:
        return "(no single shared case file; stems split — excerpts omitted)"
    p = folder / stem
    if not p.is_file():
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace").strip()
    return raw[:max_chars] + ("…" if len(raw) > max_chars else "")


def _strip_code_fence(s: str) -> str:
    """Return inner ``` / ```json block if present; do not use brace-greedy patterns (breaks nested JSON)."""
    s = s.strip()
    m = re.search(r"```(?:json)?\s*", s, re.I)
    if not m:
        return s
    rest = s[m.end() :]
    end = rest.rfind("```")
    if end < 0:
        return s
    return rest[:end].strip()


def _repair_trailing_commas(blob: str) -> str:
    out = blob
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r",(\s*})", r"\1", out)
        out = re.sub(r",(\s*\])", r"\1", out)
    return out


def _salvage_json_fields(s: str) -> dict[str, object]:
    """If strict JSON fails, pull decision / null fields so one bad row does not force INCOMPARABLE."""
    d: dict[str, object] = {}
    m = re.search(r'"decision"\s*:\s*"([^"]*)"', s, re.I)
    if m:
        d["decision"] = (m.group(1) or "INCOMPARABLE").strip().upper() or "INCOMPARABLE"
    else:
        m2 = re.search(r'"decision"\s*:\s*\'([^\']*)\'', s, re.I)
        d["decision"] = (m2.group(1).strip().upper() if m2 else "") or "INCOMPARABLE"
    if d["decision"] not in {"KEEP", "SUBSTITUTE", "INCOMPARABLE"}:
        d["decision"] = "INCOMPARABLE"
    for key in ("entity_a", "entity_b"):
        if re.search(rf'"{key}"\s*:\s*null\b', s, re.I):
            d[key] = None
        else:
            m2 = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', s, re.I | re.S)
            if m2:
                try:
                    d[key] = json.loads('"' + m2.group(1) + '"')
                except json.JSONDecodeError:
                    d[key] = m2.group(1).replace("\\n", " ")
    m3 = re.search(r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)"', s, re.I | re.S)
    if m3:
        try:
            d["rationale"] = json.loads('"' + m3.group(1) + '"')
        except json.JSONDecodeError:
            d["rationale"] = m3.group(1).replace("\\n", " ")[:500]
    else:
        d["rationale"] = "salvaged from non-JSON model output (see raw in run log)"
    d.setdefault("entity_a", None)
    d.setdefault("entity_b", None)
    return d


def parse_llm_json(raw: str) -> dict[str, object]:
    s = _strip_code_fence(raw)

    def try_raw(blob: str) -> dict[str, object] | None:
        blob = blob.strip()
        i = blob.find("{")
        if i < 0:
            return None
        try:
            obj, _end = json.JSONDecoder().raw_decode(blob, i)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    got = try_raw(s)
    if got is not None:
        return got

    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        chunk = s[i : j + 1]
        for candidate in (
            chunk,
            _repair_trailing_commas(chunk),
            chunk.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'"),
            _repair_trailing_commas(
                chunk.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
            ),
        ):
            got = try_raw(candidate)
            if got is not None:
                return got

    return _salvage_json_fields(s)


def call_ollama_json(user: str, *, model: str, system: str | None = None) -> dict[str, object]:
    sys_msg = SYSTEM if system is None else system
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.2, "num_predict": 512},
    )
    msg = getattr(resp, "message", None)
    content = (getattr(msg, "content", None) or "").strip()
    if not content:
        raise RuntimeError("empty model response")
    return parse_llm_json(content)


def final_entity(orig: str, proposed: object) -> str:
    if proposed is None or (isinstance(proposed, str) and not proposed.strip()):
        return orig.strip()
    return str(proposed).strip()


def load_rf2_match_bundle(rf2_root: Path):
    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")
    exact_map = load_rf2_term_to_concept(desc_files)
    rows = load_rf2_en_description_rows(desc_files)
    inv = index_description_rows(rows)
    parents = load_rf2_isa_parent_map(rel_files)
    return exact_map, rows, inv, parents


def depth_for_entity(
    entity: str,
    exact_map: dict[str, str],
    rows: list,
    inv: dict,
    parents: dict[str, list[str]],
    memo: dict[str, int | None],
) -> tuple[str | None, int | None]:
    el = entity.strip().lower()
    if not el:
        return None, None
    cid = best_snomed_match(el, exact_map, rows, inv)
    if not cid:
        return None, None
    d = max_depth_from_root_rf2(cid, parents, RF2_ROOT_CONCEPT_ID, memo)
    return cid, d


def _load_hierarchical_seed_rows(
    raw: list[dict[str, str]],
    max_rows: int | None,
    *,
    rng_seed: int = 42,
) -> list[dict[str, str]]:
    """Seeds from ``per_case_hierarchical_ner_pairs`` CSV (entity_a / entity_b, distinct strings)."""
    usable = [
        r
        for r in raw
        if (r.get("case_file") or "").strip()
        and (r.get("entity_a") or "").strip()
        and (r.get("entity_b") or "").strip()
    ]
    if not usable:
        return []
    if max_rows is not None and max_rows > 0 and len(usable) > max_rows:
        rng = random.Random(rng_seed)
        usable = rng.sample(usable, max_rows)
    out: list[dict[str, str]] = []
    for r in usable:
        stem = r["case_file"].strip()
        out.append(
            {
                "entity_a": r["entity_a"].strip(),
                "entity_b": r["entity_b"].strip(),
                "_category": "Per-case hierarchical is-a",
                "seed_case_stem": stem,
                "seed_excerpt_a": (r.get("context_sentence_a") or "").strip()[:950],
                "seed_excerpt_b": (r.get("context_sentence_b") or "").strip()[:950],
                "concept_a": (r.get("snomed_concept_a") or "").strip(),
                "concept_b": (r.get("snomed_concept_b") or "").strip(),
                "depth_a": str(r.get("depth_a", "")).strip(),
                "depth_b": str(r.get("depth_b", "")).strip(),
            }
        )
    return out


def load_per_case_seed_rows(csv_path: Path, max_rows: int | None, *, rng_seed: int = 42) -> list[dict[str, str]]:
    """Build LLM seed dicts from a per-case CSV: same-string overlap or hierarchical is-a pairs."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        return []
    keys = set(raw[0].keys())
    if "entity_a" in keys and "entity_b" in keys:
        return _load_hierarchical_seed_rows(raw, max_rows, rng_seed=rng_seed)
    filtered = [
        r for r in raw if (r.get("case_file") or "").strip() and (r.get("entity") or "").strip()
    ]
    if not filtered:
        return []
    if max_rows is not None and max_rows > 0 and len(filtered) > max_rows:
        rng = random.Random(rng_seed)
        filtered = rng.sample(filtered, max_rows)
    out: list[dict[str, str]] = []
    for r in filtered:
        stem = r["case_file"].strip()
        ent = r["entity"].strip()
        out.append(
            {
                "entity_a": ent,
                "entity_b": ent,
                "_category": "Per-case same string",
                "seed_case_stem": stem,
                "seed_excerpt_a": (r.get("context_sentence_a") or "").strip()[:950],
                "seed_excerpt_b": (r.get("context_sentence_b") or "").strip()[:950],
                "concept_a": (r.get("snomed_concept") or "").strip(),
                "concept_b": (r.get("snomed_concept") or "").strip(),
                "depth_a": (r.get("depth") or "").strip(),
                "depth_b": (r.get("depth") or "").strip(),
            }
        )
    return out


def run_llm_review(
    pairs: list[dict[str, str]],
    out_csv: Path,
    *,
    sleep_sec: float,
    model: str,
) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    for i, r in enumerate(pairs, start=1):
        ea, eb = r.get("entity_a", ""), r.get("entity_b", "")
        cat = r.get("_category", "")
        stem_out = (r.get("seed_case_stem") or "").strip()
        pre_a = (r.get("seed_excerpt_a") or "").strip()
        pre_b = (r.get("seed_excerpt_b") or "").strip()
        if pre_a and pre_b:
            ex_a, ex_b = pre_a, pre_b
        elif stem_out:
            ex_a = excerpt_for_case(stem_out, CORPUS_A)
            ex_b = excerpt_for_case(stem_out, CORPUS_B)
        else:
            stem_out = pick_case_stem(ea, eb) or ""
            if stem_out:
                ex_a = excerpt_for_case(stem_out, CORPUS_A)
                ex_b = excerpt_for_case(stem_out, CORPUS_B)
            else:
                ex_a = first_excerpt_any(CORPUS_A, ea)
                ex_b = first_excerpt_any(CORPUS_B, eb)
        hier = "hierarchical" in (cat or "").lower()
        tpl = USER_TEMPLATE_HIERARCHICAL if hier else USER_TEMPLATE
        sys_override = SYSTEM_HIERARCHICAL if hier else None
        user = tpl.format(
            case_stem=stem_out or "(no shared case stem resolved)",
            excerpt_a=ex_a or "(empty)",
            excerpt_b=ex_b or "(empty)",
            ea=ea.replace('"', "'"),
            eb=eb.replace('"', "'"),
        )
        try:
            js = call_ollama_json(user, model=model, system=sys_override)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(pairs)}] LLM error for {cat}: {e}", file=sys.stderr)
            js = {
                "decision": "INCOMPARABLE",
                "entity_a": None,
                "entity_b": None,
                "rationale": f"pipeline error: {e}",
            }
        dec = str(js.get("decision", "KEEP")).upper()
        fa = final_entity(ea, js.get("entity_a"))
        fb = final_entity(eb, js.get("entity_b"))
        rat = str(js.get("rationale", "")).replace("\n", " ")[:500]
        if stem_out:
            ctx_a = context_from_case_file(CORPUS_A, stem_out, fa)
            ctx_b = context_from_case_file(CORPUS_B, stem_out, fb)
        else:
            ctx_a = context_first_file_containing(CORPUS_A, ea)
            ctx_b = context_first_file_containing(CORPUS_B, eb)
        out_rows.append(
            {
                "category": cat,
                "case_stem": stem_out,
                "orig_entity_a": ea,
                "orig_entity_b": eb,
                "context_sentence_a": ctx_a,
                "context_sentence_b": ctx_b,
                "llm_decision": dec,
                "final_entity_a": fa,
                "final_entity_b": fb,
                "rationale": rat,
                "orig_concept_a": r.get("concept_a", ""),
                "orig_concept_b": r.get("concept_b", ""),
                "orig_depth_a": r.get("depth_a", ""),
                "orig_depth_b": r.get("depth_b", ""),
            }
        )
        print(f"[{i}/{len(pairs)}] {cat}  {dec}  {fa!r} vs {fb!r}", flush=True)
        if sleep_sec > 0 and i < len(pairs):
            time.sleep(sleep_sec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(out_rows[0].keys()) if out_rows else []
    if keys:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        print(f"Wrote {out_csv}", file=sys.stderr)
    return out_rows


def run_depth_pass(
    review_rows: list[dict[str, str]],
    out_depth: Path,
    rf2_root: Path,
) -> None:
    exact_map, rows, inv, parents = load_rf2_match_bundle(rf2_root)
    memo: dict[str, int | None] = {}
    depth_rows: list[dict[str, object]] = []
    for r in review_rows:
        dec = r.get("llm_decision", "KEEP")
        fa = (r.get("final_entity_a") or "").strip()
        fb = (r.get("final_entity_b") or "").strip()
        ca, da = depth_for_entity(fa, exact_map, rows, inv, parents, memo)
        cb, db = depth_for_entity(fb, exact_map, rows, inv, parents, memo)
        depth_rows.append(
            {
                **r,
                "snomed_concept_a": ca or "",
                "snomed_concept_b": cb or "",
                "depth_a": "" if da is None else da,
                "depth_b": "" if db is None else db,
                "depth_diff_b_minus_a": ""
                if da is None or db is None
                else int(db) - int(da),
            }
        )
    out_depth.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen_k: set[str] = set()
    for row in depth_rows:
        for k in row:
            if k not in seen_k:
                seen_k.add(k)
                keys.append(k)
    if keys:
        with out_depth.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in depth_rows:
                w.writerow({k: row.get(k, "") for k in keys})
    print(f"Wrote {out_depth}", file=sys.stderr)


def load_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ensure_results()
    p = argparse.ArgumentParser(
        description=(
            "LLM review of NER entity pairs, then SNOMED depth on final strings. "
            "Default seeds: one divergent row per SNOMED top-level category from branch_pairs. "
            "Alternative: --seeds-csv from per_case_same_string_ner (same surface string) or "
            "per_case_hierarchical_ner_pairs (different strings, SNOMED is-a broad vs specific)."
        ),
        epilog=(
            "Mistral-free paths: use --depth-only to recompute SNOMED depths from an existing "
            "review CSV (no Ollama). To refresh taxonomy JPEGs from existing depths, run "
            "python -m pipeline.llm_taxonomy_viz (no LLM)."
        ),
    )
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument("--branch-pairs", type=Path, default=SNOMED_BRANCH_PAIRS)
    p.add_argument(
        "--seeds-csv",
        type=Path,
        default=None,
        help=(
            "Per-case seeds CSV: either per_case_same_string_ner output (column ``entity``) or "
            "per_case_hierarchical_ner_pairs output (columns ``entity_a``, ``entity_b``). "
            "When set, branch_pairs is not read; use --max-llm-rows to cap Ollama calls."
        ),
    )
    p.add_argument(
        "--max-llm-rows",
        type=int,
        default=80,
        metavar="N",
        help="With --seeds-csv: random sample up to N rows (seed fixed). Use 0 for all rows.",
    )
    p.add_argument("--review-out", type=Path, default=None)
    p.add_argument("--depth-out", type=Path, default=None)
    p.add_argument("--min-abs-depth-diff", type=int, default=1)
    p.add_argument(
        "--max-branch-rows",
        type=int,
        default=None,
        help="Cap rows when scanning branch_pairs (same as snomed_taxonomy_compare).",
    )
    p.add_argument("--sleep", type=float, default=SLEEP_SEC, help="Pause between Ollama calls.")
    p.add_argument("--model", type=str, default=MODEL, help="Ollama model name.")
    p.add_argument(
        "--depth-only",
        action="store_true",
        help="Skip Ollama; read --review-in and only write SNOMED depths.",
    )
    p.add_argument(
        "--review-in",
        type=Path,
        default=None,
        help="With --depth-only, CSV from a previous run (default: --review-out).",
    )
    args = p.parse_args()

    use_per_case = args.seeds_csv is not None

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))

    if args.depth_only:
        review_out = args.review_out or (
            LLM_PAIR_REVIEW_PER_CASE if use_per_case else LLM_PAIR_REVIEW_CSV
        )
        depth_out = args.depth_out or (
            LLM_ALIGNED_DEPTHS_PER_CASE if use_per_case else LLM_ALIGNED_DEPTHS_CSV
        )
        src = args.review_in or review_out
        if not src.is_file():
            raise SystemExit(f"Missing review CSV: {src}")
        rows = load_review_csv(src)
        run_depth_pass(rows, depth_out, rf2_root)
        return

    seeds: list[dict[str, str]] = []
    if use_per_case:
        seeds_path = args.seeds_csv.resolve()
        if not seeds_path.is_file():
            raise SystemExit(f"Missing seeds CSV: {seeds_path}")
        cap = None if args.max_llm_rows == 0 else args.max_llm_rows
        seeds = load_per_case_seed_rows(seeds_path, cap)
        if not seeds:
            raise SystemExit(f"No rows in {seeds_path}")
        hier = seeds[0].get("_category") == "Per-case hierarchical is-a"
        review_out = args.review_out or (
            LLM_PAIR_REVIEW_PER_CASE_HIERARCHICAL if hier else LLM_PAIR_REVIEW_PER_CASE
        )
        depth_out = args.depth_out or (
            LLM_ALIGNED_DEPTHS_PER_CASE_HIERARCHICAL if hier else LLM_ALIGNED_DEPTHS_PER_CASE
        )
        kind = "hierarchical is-a" if hier else "same-string"
        print(f"LLM review for {len(seeds)} per-case {kind} seed(s) (cap={cap}).", file=sys.stderr)
    else:
        if not args.branch_pairs.is_file():
            raise SystemExit(f"Missing {args.branch_pairs}. Run pipeline.branch_pairs first.")

        _, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
        if not rel_files:
            raise SystemExit("No RF2 relationship files.")
        parents = load_rf2_isa_parent_map(rel_files)
        seeds = pick_divergent_pairs_per_category(
            args.branch_pairs, parents, args.min_abs_depth_diff, args.max_branch_rows
        )
        if not seeds:
            raise SystemExit("No divergent seed pairs; loosen --min-abs-depth-diff.")

        print(f"LLM review for {len(seeds)} seed pair(s) (one per SNOMED top-level group).", file=sys.stderr)
        review_out = args.review_out or LLM_PAIR_REVIEW_CSV
        depth_out = args.depth_out or LLM_ALIGNED_DEPTHS_CSV

    reviewed = run_llm_review(seeds, review_out, sleep_sec=args.sleep, model=args.model)
    run_depth_pass(reviewed, depth_out, rf2_root)


if __name__ == "__main__":
    main()
