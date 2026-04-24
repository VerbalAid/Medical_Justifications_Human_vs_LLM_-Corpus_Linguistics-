# Corpus linguistics pipeline (CasiMedicos human vs Mistral)

> **Layout:** This folder is **`register_audit_mcq/`** inside the parent **Terminology** workspace. Use **`../.venv`** for the virtualenv and **`../SnomedCT_.../`** for SNOMED RF2 (see `pipeline/paths.py`: `DEFAULT_RF2_ROOT` points at the parent directory). Run `python -m pipeline.<module>` from the **parent** `Terminology` directory **or** from this folder (both work if the venv is active).

Compare **human clinical explanations** with **locally generated LLM explanations** on the same 125 English test cases from [HiTZ/casimedicos-arg](https://huggingface.co/datasets/HiTZ/casimedicos-arg), then run frequency, TF–IDF, and optional SNOMED CT analysis.

Run everything from the **Terminology** parent or this folder with `python -m pipeline.<module>` (see table below). Code lives under `pipeline/`; CSVs and plots go to `results/` by default.

A full write-up that pulls numbers and figures from `results/` is in **`report/`** (`main.tex` → `main.pdf`; see `report/README.txt` to compile).

---

## How to run everything (quick)

**You need:** cwd = project root · **Python 3.12** venv (`.venv`) · `pip install -r requirements.txt` · **Ollama** + `ollama pull mistral` · **SNOMED RF2** unpacked (`export RF2=/path/to/.../` unless the default folder name already sits next to this repo — see `pipeline/paths.py`).

```bash
cd "/path/to/Terminology"
source .venv/bin/activate
export RF2="/path/to/SnomedCT_InternationalRF2_PRODUCTION_.../"

# Corpora + lexical stats (no RF2)
python -m pipeline.prep_corpora
python -m pipeline.prune_corpora_pairs
python -m pipeline.freq_zipf && python -m pipeline.hedging && python -m pipeline.hedging_area && python -m pipeline.tfidf

# SNOMED + NER + branch pairs + taxonomy JPEGs (one chain after rf2)
python -m pipeline.rf2 --rf2-root "$RF2"
python -m pipeline.run_snomed_ner_chain --rf2-root "$RF2"
# NER CSV already there? use: ... --skip-ner
# Smaller subgraph in that chain? add: --graph-compact

# Optional — LLM review + same-style taxonomy JPEGs for aligned strings (Ollama running)
python -m pipeline.llm_pair_align --rf2-root "$RF2"
# python -m pipeline.llm_pair_align --depth-only --rf2-root "$RF2"   # no Ollama; refresh depths from saved review CSV
python -m pipeline.llm_taxonomy_viz --rf2-root "$RF2"

# Report (must run inside report/ — see report/README.txt)
cd report && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex main && pdflatex main && cd ..
```

**Outputs:** `corpus_a/`, `corpus_b/` · `results/` (table below) · `report/main.pdf`. **Extra copy-paste / flags:** `docs/command_prompts.md`. **Proposal vs repo (AntConc gaps, unkeyed B):** `docs/proposal_alignment.md`.

---

## What each step does

### 1. `prep_corpora`

Loads the **English test** split. For each case:

- **Corpus A (human):** text **after** the line containing **CORRECT ANSWER:** (experts’ justifications).
- **Corpus B (model):** the same case up to (but not including) **CORRECT ANSWER:** is sent to **Mistral** via Ollama. The prompt states the **correct option** and asks for a **teaching-style clinical explanation** (not MCQ elimination), with a **word-count target** derived from the human justification on that row so length is comparable when possible.

Outputs: `corpus_a/case_*.txt` and `corpus_b/case_*.txt`. Use `--force` to regenerate all pairs after you change the prompt logic. Default generation is **keyed-correct** (model told the right option). For **proposal-style** Corpus B (stem + five options only, no revealed answer), use **`--omit-correct-option`** (typically with `--force`); then re-run downstream stats.

### 2. `freq_zipf`

Token frequencies with English stopwords removed; Zipf-style log–log plot; top 50 terms per corpus under `results/corpus_frequency/` (`human_top50_token_freq.csv`, `model_top50_token_freq.csv`, `zipf_rank_frequency_loglog.png`). With `--rf2-root`, also writes the `*_freq_snomed_token_subset.csv` files there.

### 3. `hedging`

Counts a small **hedging / epistemic** word list in A vs B, normalised **per 1,000 tokens**, → `results/hedging/hedge_density_both_corpora.csv`.

### 4. `tfidf`

One **document per case file**; TF–IDF fitted **separately** on A and on B. Shared terms with mean weight gap (A − B) → `results/tfidf/shared_terms_mean_tfidf_gap.csv`.

### 5. `rf2` (SNOMED depth from frequency lists)

Top terms from the frequency CSVs; exact match on RF2 **Description** strings; **is-a** depth to root → `results/snomed/snomed_depth_from_freq_top_terms.csv`.

### 6. `ner_depth` (optional)

**scispaCy** `en_core_sci_lg` entities; exact then substring match to descriptions; same depth rule → `results/snomed/snomed_depth_from_ner_entities.csv`. Compares mean depth to the freq-based depth CSV when that file exists.

**Python:** use a **3.12** venv for NER (`python3.12 -m venv .venv`). On Fedora, a default **3.14** venv often breaks spaCy/numpy wheels. Model URL: **ai2-s2-scispacy** (see `requirements.txt`).

### 7. Other SNOMED helpers (optional)

| Module | Role |
|--------|------|
| `graph_viz` | Subgraph PNG from NER depth CSV → `results/snomed/plot_subgraph_high_depth_ner.png`. Use `--compact` or `--ancestor-hops N` for a readable slice; full closure is often a hairball. |
| `pair_depths` | Same entity string in A and B → `results/snomed/paired_entity_depth_same_string.csv`, `hist_depth_difference_b_minus_a.png` |
| `branch_pairs` | A×B branch pairs (LCA, hop limit); `results/snomed/branch_pairs_all_concepts.csv`; `branch_pairs_category_counts.csv` |
| `snomed_taxonomy_compare` | One divergent pair per SNOMED top-level category → JPEGs in `results/snomed/taxonomy_compare_by_category_jpeg/` plus `taxonomy_compare_overview_strip.jpg`. Options: `--jpeg-dir`, `--overview-jpeg`, `--max-rows N`. |
| `llm_pair_align` | **Optional:** same divergent seeds as taxonomy → Ollama reads same-case excerpts + NER pair → JSON (`KEEP` / `SUBSTITUTE` / `INCOMPARABLE`) → `results/llm_align/llm_pair_review.csv` and `llm_aligned_snomed_depths.csv` (both include `context_sentence_a` / `context_sentence_b`: the corpus sentence around each original NER string). `--depth-only` recomputes depths from an existing review CSV. |
| `llm_taxonomy_viz` | After `llm_aligned_snomed_depths.csv` exists: same two-column JPEG layout as `snomed_taxonomy_compare`, but for **LLM-final** concepts/depths → `results/llm_align/taxonomy_jpeg/*.jpg` + `llm_taxonomy_overview_strip.jpg`. Embeds **corpus sentence context** under each pair (from CSV columns or recomputed from `orig_entity_*` + `case_stem`). Rows without both concepts and depths are skipped. |

All SNOMED steps need an unzipped International **RF2** tree; pass `--rf2-root` if it is not the default folder name next to the project (see `pipeline/paths.py`).

**Branch pairs vs taxonomy JPEGs:** `branch_pairs` compares **every** NER span from human texts against **every** NER span from model texts (full cross-product, subject to LCA rules). The strings do **not** need to match. `snomed_taxonomy_compare` then picks **one** row per SNOMED top-level category — the pair with the largest depth difference — as a **picture-friendly example**, not as proof that those two phrases are “the same” clinical object. For **same surface string** in A and B, use `pair_depths` instead.

---

## Setup

```bash
cd "/path/to/Terminology"
pip install -r requirements.txt
```

- **Ollama:** [ollama.com](https://ollama.com), then `ollama pull mistral`.
- **Hugging Face:** first run sets `HF_HOME` to `.hf_cache` under the project.
- **SNOMED (optional):** unpack International RF2; pass `--rf2-root` to that folder.

### Dataset quirk

Some English **test** rows have `text: null` or otherwise unusable `text` in the published table. `prep_corpora` **filters them out**: no `case_NNN.txt` for those indices (stale pairs are deleted); Ollama is not called. A short summary counts kept vs filtered rows at the end of the run.

---

## Run order

`prep_corpora` **does not call Mistral** for a case when `corpus_b/case_NNN.txt` already exists and is non-empty (Ollama stays off for that file). It still refreshes `corpus_a` from the dataset. Use **`--force`** to regenerate **all** model files.

```bash
python -m pipeline.prep_corpora
# python -m pipeline.prep_corpora --force

python -m pipeline.prune_corpora_pairs

python -m pipeline.freq_zipf
python -m pipeline.hedging
python -m pipeline.hedging_area
python -m pipeline.tfidf

python -m pipeline.rf2 --rf2-root "$RF2"
python -m pipeline.ner_depth --rf2-root "$RF2"
python -m pipeline.graph_viz --rf2-root "$RF2"          # add --compact for a smaller figure
python -m pipeline.pair_depths
python -m pipeline.branch_pairs --rf2-root "$RF2" --dedupe-concept-pairs
python -m pipeline.snomed_taxonomy_compare --rf2-root "$RF2"   # optional: --max-rows 200000
```

**One command** after `rf2` (same order as the block above; needs **Python 3.12** for spaCy):

```bash
python -m pipeline.run_snomed_ner_chain --rf2-root "$RF2"
```

If `snomed_depth_from_ner_entities.csv` already exists:

```bash
python -m pipeline.run_snomed_ner_chain --rf2-root "$RF2" --skip-ner
```

---

## Outputs

| Location | Produced by |
|----------|-------------|
| `corpus_a/`, `corpus_b/` | `prep_corpora` |
| `results/corpus_frequency/*` (freq CSVs + Zipf PNG) | `freq_zipf` |
| `results/hedging/hedge_density_both_corpora.csv` | `hedging` |
| `results/hedging/specialty_lexicon_tagging/*` (per-case + group tables + bar PNGs) | `hedging_area` |
| `results/tfidf/shared_terms_mean_tfidf_gap.csv` | `tfidf` |
| `results/snomed/snomed_depth_from_freq_top_terms.csv` | `rf2` |
| `results/snomed/snomed_depth_from_ner_entities.csv` | `ner_depth` |
| `results/snomed/plot_subgraph_high_depth_ner.png` | `graph_viz` |
| `results/snomed/paired_entity_depth_same_string.csv`, `hist_depth_difference_b_minus_a.png` | `pair_depths` |
| `results/snomed/per_case_same_string_entity_overlap.csv` | `per_case_same_string_ner` (mention-aligned identical strings per twin) |
| `results/snomed/per_case_hierarchical_entity_pairs.csv` | `per_case_hierarchical_ner_pairs` (different strings, strict `is_a` + depth gap; vagueness-oriented slice) |
| `results/snomed/branch_pairs_all_concepts.csv`, `branch_pairs_category_counts.csv` | `branch_pairs` |
| `results/snomed/taxonomy_compare_by_category_jpeg/*.jpg`, `taxonomy_compare_overview_strip.jpg` | `snomed_taxonomy_compare` |
| `results/llm_align/*.csv`, `taxonomy_jpeg/*.jpg`, `llm_taxonomy_overview_strip.jpg` | `llm_pair_align`, `llm_taxonomy_viz` |

Command-line cheat sheet: `docs/command_prompts.md`. **Figure interpretation + AntConc** (keywords, collocations, KWIC): `docs/antconc_qualitative/README.md`.

**Report figures** (TF--IDF gap bars + same-string and hierarchical LLM bar charts, and `report/_generated_llm_hier.tex` counts for `main.tex`): after `tfidf` and the per-case LLM CSVs exist, run `python3 report/gen_report_figures.py` → `results/report_figures/*.png` plus the generated TeX snippet.

---

## Licence / data

- **CasiMedicos-Arg** is **CC-BY 4.0** (dataset card on Hugging Face).
- **SNOMED CT** use is subject to your **SNOMED International** licence; RF2 files are not redistributed here.
