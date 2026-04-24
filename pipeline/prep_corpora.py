#!/usr/bin/env python3
# Load CasiMedicos English test from Hugging Face: human key -> corpus_a, model via Ollama -> corpus_b.

from __future__ import annotations
import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path

import ollama
from datasets import load_dataset

from .paths import CORPUS_A, CORPUS_B, HF_CACHE, ROOT
from .prune_corpora_pairs import prune_paired_corpora

MODEL = "mistral"
MAX_ATTEMPTS = 3
RETRY_SLEEP_SEC = 2.0


def _file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0

SYSTEM_PROMPT = (
    "You are a clinician writing short teaching explanations in British English for expert readers. "
    "No roleplay, no 'as an AI', no generic disclaimers. The user gives one case: the answer is already fixed; "
    "write only that explanation. Keep length similar to a normal exam key style paragraph, not an essay."
)

SYSTEM_PROMPT_UNKEYED = (
    "You are a clinician writing short explanations in British English for expert readers. "
    "No roleplay, no 'as an AI', no generic disclaimers. The user gives an MCQ vignette and five options but "
    "does **not** state which option is correct: argue for the diagnosis you judge best fits, in compact prose."
)


def parse_tokenized_text(text_field: object) -> list[list[str]]:
    if text_field is None:
        raise ValueError("text field is None")
    if isinstance(text_field, list):
        if not text_field:
            return []
        if not isinstance(text_field[0], list):
            raise ValueError("expected text as list of token lists")
        return text_field
    if not isinstance(text_field, str):
        text_field = str(text_field)
    s = text_field.strip()
    if s in ("", "None", "null"):
        raise ValueError("text empty or null string")
    parsed = ast.literal_eval(s)
    if parsed is None or not isinstance(parsed, list):
        raise ValueError("text did not parse to a list")
    return parsed


def sentences_to_string(sentences: list[list[str]]) -> str:
    return "\n\n".join(" ".join(toks) for toks in sentences)


def find_correct_answer_index(sentences: list[list[str]]) -> int:
    for i, toks in enumerate(sentences):
        if re.search(r"\bCORRECT\s+ANSWER\s*:", " ".join(toks), flags=re.IGNORECASE):
            return i
    raise ValueError("CORRECT ANSWER: line not found")


def extract_human_justification(sentences: list[list[str]]) -> str:
    idx = find_correct_answer_index(sentences)
    tail = sentences[idx + 1 :]
    return sentences_to_string(tail) if tail else ""


def parse_correct_option_number(sentences: list[list[str]]) -> int:
    idx = find_correct_answer_index(sentences)
    line = " ".join(sentences[idx])
    m = re.search(r"\bCORRECT\s+ANSWER\s*:\s*(\d+)", line, flags=re.IGNORECASE)
    if not m:
        raise ValueError("no option digit after CORRECT ANSWER:")
    n = int(m.group(1))
    if n < 1 or n > 5:
        raise ValueError(f"option out of range 1–5: {n}")
    return n


def find_option_label(sentences: list[list[str]], correct_line_idx: int, option_num: int) -> str:
    pat = re.compile(rf"^{option_num}\s*-\s*(.+)$")
    for toks in sentences[:correct_line_idx]:
        line = " ".join(toks).strip()
        mo = pat.match(line)
        if mo:
            return mo.group(1).strip()
    return ""


def _length_constraints(human_ref: str) -> tuple[str, int]:
    human_words = len(human_ref.split()) if human_ref.strip() else 0
    if human_words < 1:
        text = (
            "Keep to one **short** expert paragraph (roughly **80–120 words**); no bullet laundry lists "
            "or long differential walkthroughs. "
        )
        return text, 384
    max_words = max(human_words + 8, int(human_words * 1.28))
    num_predict = max(180, min(int(human_words * 1.22 * 1.35) + 48, 1536))
    text = (
        f"The paired expert justification for this item is about **{human_words}** words. "
        f"Match that scope: **about {human_words}–{max_words} words**, tight prose—**do not** pad, "
        "recap all five options, or write several times longer than the expert would. "
        "No long enumerated differentials unless strictly necessary. "
    )
    return text, num_predict


def build_generation_prompt(sentences: list[list[str]], *, human_ref: str = "") -> tuple[str, int]:
    correct_idx = find_correct_answer_index(sentences)
    option_num = parse_correct_option_number(sentences)
    label = find_option_label(sentences, correct_idx, option_num)
    head = sentences[:correct_idx]
    body = sentences_to_string(head)
    keyed = f"option {option_num} ({label})" if label else f"option {option_num}"
    length_hint, num_predict = _length_constraints(human_ref)
    instruction = (
        "You are writing a **clinical explanation** for colleagues, not doing multiple-choice elimination. "
        "Below are the case, question, and five options; the diagnosis below is **already the correct one**—"
        "your job is to **elaborate** why it fits: relevant findings, pathophysiology or clinical logic, "
        "and how the presentation matches. Use an explanatory, teaching tone. "
        f"The diagnosis to explain is **{keyed}**. "
        f"{length_hint}"
        + "Mention alternatives only briefly if it clarifies this diagnosis. Do not output only the option number."
    )
    return f"{instruction}\n\n{body}", num_predict


def build_generation_prompt_unkeyed(sentences: list[list[str]], *, human_ref: str = "") -> tuple[str, int]:
    """Proposal-style prompt: stem + options only (no revealed correct answer)."""
    correct_idx = find_correct_answer_index(sentences)
    head = sentences[:correct_idx]
    body = sentences_to_string(head)
    length_hint, num_predict = _length_constraints(human_ref)
    instruction = (
        "You are writing a **clinical explanation** for colleagues based on the case stem and the **five** "
        "listed answer options. **You are not told** which option is the official correct answer. "
        "Argue for the diagnosis you judge best supported by the vignette, in a teaching tone. "
        f"{length_hint}"
        "Do not output only an option number; use connected prose. You may mention alternatives briefly "
        "if it strengthens your preferred reading."
    )
    return f"{instruction}\n\n{body}", num_predict


def call_ollama(user_prompt: str, *, num_predict: int, system: str | None = None) -> str:
    sys_msg = system if system is not None else SYSTEM_PROMPT
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_predict": num_predict},
    )
    msg = getattr(resp, "message", None)
    if msg is None:
        raise RuntimeError("Ollama response missing message")
    content = (getattr(msg, "content", None) or "").strip()
    if not content:
        raise RuntimeError("Ollama returned empty content")
    return content


def remove_case_outputs(path_a: Path, path_b: Path) -> None:
    for p in (path_a, path_b):
        if p.is_file():
            p.unlink()


def preprocess_row_text(raw_text: object) -> tuple[list[list[str]] | None, str | None]:
    if raw_text is None:
        return None, "missing text"
    try:
        return parse_tokenized_text(raw_text), None
    except (ValueError, SyntaxError) as e:
        return None, f"unparseable text ({e})"


def generate_with_retries(prompt: str, *, num_predict: int, system: str | None = None) -> str:
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call_ollama(prompt, num_predict=num_predict, system=system)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [warn] attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SEC * attempt)
    assert last is not None
    raise last


def load_subset_case_nums(path: Path) -> set[int]:
    """Lines like ``case_001.txt``, ``001``, or ``1`` → case indices (1-based)."""
    out: set[int] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("case_") and line.lower().endswith(".txt"):
            stem = line[5:-4].lstrip("0") or "0"
            out.add(int(stem))
            continue
        digits = "".join(ch for ch in line if ch.isdigit())
        if digits:
            out.add(int(digits))
    if not out:
        raise ValueError(f"No case indices parsed from {path}")
    return out


def main(
    *,
    force: bool = False,
    omit_correct_option: bool = False,
    subset_path: Path | None = None,
    out_corpus_b: Path | None = None,
) -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    CORPUS_A.mkdir(parents=True, exist_ok=True)
    CORPUS_B.mkdir(parents=True, exist_ok=True)

    subset_nums: set[int] | None = None
    if subset_path is not None:
        subset_nums = load_subset_case_nums(subset_path.resolve())
        print(f"Subset mode: {len(subset_nums)} case index(es) from {subset_path}", flush=True)

    corp_b_root = Path(out_corpus_b).resolve() if out_corpus_b is not None else CORPUS_B
    if out_corpus_b is not None:
        corp_b_root.mkdir(parents=True, exist_ok=True)
        print(f"Corpus B output directory: {corp_b_root}", flush=True)

    print("Loading HiTZ/casimedicos-arg (config=en, split=test)...")
    ds = load_dataset("HiTZ/casimedicos-arg", "en", split="test")
    n = len(ds)
    if n != 125:
        print(f"Warning: expected 125 test cases, got {n}", file=sys.stderr)

    filtered = 0
    for i in range(n):
        case_num = i + 1
        if subset_nums is not None and case_num not in subset_nums:
            continue
        out_name = f"case_{case_num:03d}.txt"
        path_a = CORPUS_A / out_name
        path_b = corp_b_root / out_name

        # Fast path: nothing to do when both sides already look good.
        if not force and _file_nonempty(path_a) and _file_nonempty(path_b):
            print(f"[{case_num}/{n}] skip (corpus A and B already present)")
            continue

        row = ds[i]
        rid = row.get("id", case_num)
        raw_text = row["text"]
        sents, skip_reason = preprocess_row_text(raw_text)
        if skip_reason is not None:
            remove_case_outputs(path_a, path_b)
            filtered += 1
            print(
                f"[{case_num}/{n}] filtered out: {skip_reason} (id={rid}); no case files",
                file=sys.stderr,
            )
            continue

        human = extract_human_justification(sents)
        path_a.write_text(human, encoding="utf-8")

        if not force and _file_nonempty(path_b):
            print(f"[{case_num}/{n}] {out_name}  skip Mistral (corpus B already done)", flush=True)
            continue

        try:
            if omit_correct_option:
                prompt, num_predict = build_generation_prompt_unkeyed(sents, human_ref=human)
                sys_use = SYSTEM_PROMPT_UNKEYED
            else:
                prompt, num_predict = build_generation_prompt(sents, human_ref=human)
                sys_use = None
        except ValueError as e:
            print(f"[{case_num}/{n}] warn: cannot build prompt ({e}); empty corpus B", file=sys.stderr)
            path_b.write_text("", encoding="utf-8")
            continue

        llm_text = generate_with_retries(prompt, num_predict=num_predict, system=sys_use)
        path_b.write_text(llm_text, encoding="utf-8")
        print(
            f"[{case_num}/{n}] {out_name}  A={len(human)} chars  B={len(llm_text)} chars  ({MODEL})",
            flush=True,
        )

    kept = n - filtered
    print(f"Done. {kept}/{n} rows kept for corpus files ({filtered} filtered out).")
    if subset_nums is None and out_corpus_b is None:
        pairs_kept, stems_pruned = prune_paired_corpora()
        print(
            f"Paired corpora: {pairs_kept} strict non-empty A/B pair(s); "
            f"removed {stems_pruned} incomplete stem(s).",
            flush=True,
        )
    else:
        print("Subset or alternate Corpus B dir: skipped global prune_paired_corpora().", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build corpus_a and corpus_b from CasiMedicos-Arg EN test.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every case, including re-calling Mistral for corpus B.",
    )
    p.add_argument(
        "--omit-correct-option",
        action="store_true",
        help=(
            "Proposal-style Corpus B: prompt includes stem and five options only (no revealed correct answer). "
            "Uses a separate system prompt. Implies re-generation of B unless files exist and --force is not set."
        ),
    )
    p.add_argument(
        "--subset",
        type=Path,
        default=None,
        metavar="FILE",
        help="Only process 1-based case indices listed in FILE (one per line: case_001.txt or 001).",
    )
    p.add_argument(
        "--out-corpus-b",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write corpus B files under this directory instead of corpus_b/ (use with --subset for ablations).",
    )
    args = p.parse_args()
    main(
        force=args.force,
        omit_correct_option=args.omit_correct_option,
        subset_path=args.subset,
        out_corpus_b=args.out_corpus_b,
    )
