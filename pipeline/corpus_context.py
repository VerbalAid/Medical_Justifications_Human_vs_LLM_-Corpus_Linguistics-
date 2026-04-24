"""Extract a short sentence (or clause window) around an entity string in case text."""

from __future__ import annotations

import re
from pathlib import Path

# Characters that may follow sentence-ending punctuation before whitespace.
_SKIP_AFTER_SENTENCE = frozenset("\"')]}")


def find_entity_span(text: str, entity: str) -> tuple[int, int] | None:
    """Return (start, end) indices in ``text`` for the first match of ``entity``."""
    e = (entity or "").strip()
    if not e:
        return None
    tl = text.lower()
    el = e.lower()
    if " " in el or "-" in el or len(el) < 2:
        m = re.search(re.escape(e), text, re.I)
        if not m:
            return None
        return m.start(), m.end()
    m = re.search(r"(?<![a-z0-9])" + re.escape(el) + r"(?![a-z0-9])", tl, re.I)
    if not m:
        return None
    return m.start(), m.end()


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) to a rough sentence using . ! ? followed by space or EOL."""
    lo, hi = 0, len(text)
    i = start - 1
    while i >= 0:
        ch = text[i]
        if ch in ".!?":
            tail = i + 1
            while tail < len(text) and text[tail] in _SKIP_AFTER_SENTENCE:
                tail += 1
            if tail >= len(text) or text[tail] in " \n\t\r":
                lo = i + 1
                while lo < len(text) and text[lo] in " \n\t\r":
                    lo += 1
                break
        i -= 1
    j = end
    while j < len(text):
        if text[j] in ".!?":
            tail = j + 1
            while tail < len(text) and text[tail] in _SKIP_AFTER_SENTENCE:
                tail += 1
            if tail >= len(text) or text[tail] in " \n\t\r":
                hi = tail
                break
        j += 1
    return lo, hi


def surrounding_sentence(text: str, entity: str, max_chars: int = 320) -> str:
    """One sentence (or bounded window) containing ``entity``; empty if not found."""
    span = find_entity_span(text, entity)
    if not span:
        return ""
    lo, hi = _sentence_bounds(text, span[0], span[1])
    s = text[lo:hi].strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "\u2026"
    return s


def context_from_case_file(folder: Path, case_stem: str, entity: str, max_chars: int = 320) -> str:
    if not case_stem or "|" in case_stem:
        return ""
    p = folder / case_stem
    if not p.is_file():
        return ""
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return surrounding_sentence(raw, entity, max_chars=max_chars)


def context_first_file_containing(folder: Path, entity: str, max_chars: int = 320) -> str:
    """When no shared stem: first case file in ``folder`` that contains ``entity``."""
    e = (entity or "").strip()
    if not e:
        return ""
    for p in sorted(folder.glob("case_*.txt")):
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if find_entity_span(raw, e):
            return surrounding_sentence(raw, e, max_chars=max_chars)
    return ""
