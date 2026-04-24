# Command-line reference

British English in prose where it matters. **Project root for this repo** is the `register_audit_mcq/` folder (inside the parent Terminology workspace). Run commands from there, or set `PYTHONPATH` to that folder. Example: `cd …/Terminology/register_audit_mcq`.

**Common mistakes:** (1) The venv lives in the **parent** `Terminology/` folder: from `register_audit_mcq/` use `source ../.venv/bin/activate`. (2) `ModuleNotFoundError: No module named 'pipeline'` — run `python -m pipeline…` with cwd set to **`register_audit_mcq/`** (or add that folder to `PYTHONPATH`). All `results/...` paths are relative to `register_audit_mcq/`.

---

## One-off setup

```bash
cd "/path/to/Terminology/register_audit_mcq"
source ../.venv/bin/activate
pip install -r requirements.txt
ollama pull mistral
```

Start Ollama before `prep_corpora`. Hugging Face cache defaults to `./.hf_cache` on first prep run.

---

## Modules (run from project root)

| Step | Command | Output |
|------|---------|--------|
| Corpus build | `python -m pipeline.prep_corpora` (`--force`, `--omit-correct-option` for proposal-style unkeyed B; `--subset FILE`, `--out-corpus-b DIR` for stratified ablation) | `corpus_a/`, `corpus_b/` |
| Word-count summary | `python -m pipeline.corpus_length_stats` (`--per-file` for each case) | stdout (mean/median words per file for A and B) |
| Pair prune | `python -m pipeline.prune_corpora_pairs` | Drops any `case_*.txt` without a non-empty twin |
| Frequencies / Zipf | `python -m pipeline.freq_zipf` | `results/corpus_frequency/` (top-50 CSVs, Zipf PNG); optional `--rf2-root` for SNOMED-filter CSVs |
| Hedging | `python -m pipeline.hedging` | `results/hedging/hedge_density_both_corpora.csv` |
| Hedging + specialty guess | `python -m pipeline.hedging_area` | `results/hedging/specialty_lexicon_tagging/` (per-case CSV, group means, two bar PNGs); `--min-area-cases 3` |
| TF–IDF | `python -m pipeline.tfidf` | `results/tfidf/shared_terms_mean_tfidf_gap.csv` |
| SNOMED depth (freq) | `python -m pipeline.rf2 --rf2-root "..."` | `results/snomed/snomed_depth_from_freq_top_terms.csv` |
| SNOMED depth (NER) | `python -m pipeline.ner_depth` | `results/snomed/snomed_depth_from_ner_entities.csv` (needs **Python 3.12** venv + `en_core_sci_lg`) |
| Subgraph plot | `python -m pipeline.graph_viz` (add `--compact` or e.g. `--ancestor-hops 12 --top 5 --label-seeds-only` for a small readable graph) | `results/snomed/plot_subgraph_high_depth_ner.png` |
| Paired depths | `python -m pipeline.pair_depths` | `results/snomed/paired_entity_depth_same_string.csv`, `hist_depth_difference_b_minus_a.png` (pooled corpus intersection; not per-case) |
| Per-case same-string NER overlap | `python -m pipeline.per_case_same_string_ner --rf2-root "$RF2"` | `results/snomed/per_case_same_string_entity_overlap.csv` (after `ner_depth`; strict twins + entity string in both A and B for that case, with short contexts) |
| Per-case hierarchical NER (broad vs specific, is-a) | `python -m pipeline.per_case_hierarchical_ner_pairs --rf2-root "$RF2" --min-depth-gap 4 --max-pairs-per-case 15` | `results/snomed/per_case_hierarchical_entity_pairs.csv` |
| Per-case **A vs B** SNOMED (full NER inventories + set comparison) | `python -m pipeline.per_case_snomed_ab_comparison --rf2-root "$RF2"` | `per_case_ab_snomed_summary.csv` (one row per case: concept overlap, Jaccard, weighted mean depths), `per_case_ab_snomed_entity_inventory.csv` (every NER type × side × case), `per_case_ab_snomed_cross_concepts.csv` (optional capped is-a pairs across sides; `--no-cross-concepts` to skip) |
| Per-case AB figures | `python -m pipeline.per_case_ab_snomed_viz` | `per_case_ab_snomed_dashboard.png` (six summary panels), `per_case_ab_snomed_cross_pairs.png` (depth-gap + broader-side counts from cross CSV) |
| Two-case SNOMED neighbourhood (GraphML + PNG) | `python -m pipeline.snomed_two_case_neighborhood case_001.txt case_005.txt --rf2-root "$RF2"` | `results/snomed/snomed_two_case_neighborhood_001_005.graphml` (Gephi / Neo4j APOC) + `.png`; tune `--up-hops`, `--down-hops`, `--max-nodes` (SNOMED child fan-out is large). |
| Model explanation as a graph (sentences → NER → SNOMED) | `python -m pipeline.model_explanation_snomed_graph case_005.txt` | `model_explanation_graph_005_B.graphml` + `model_explanation_graph_005_B.png`. Add `--compare-a` for side-by-side human vs model PNG + separate GraphML for A. |
| Graph-RAG-style judge (LLM on trimmed graph JSON) | `python -m pipeline.model_explanation_graph_rag_judge case_005.txt` (`--rf2-root "$RF2"`, `--model mistral`) | `results/llm_align/model_explanation_graph_judge_005.json` (task text, excerpts, trimmed `graph_bundle`, parsed `judge` object). **`--dry-run`:** same file with `judge.skipped` and no Ollama call. **`--include-human`:** adds human graph when `corpus_a/` twin exists. **`--out-json PATH`** to override output. |
| Branch pairs | `python -m pipeline.branch_pairs` | `results/snomed/branch_pairs_all_concepts.csv`, `branch_pairs_category_counts.csv` |
| Taxonomy compare | `python -m pipeline.snomed_taxonomy_compare` | `results/snomed/taxonomy_compare_by_category_jpeg/*.jpg`, `taxonomy_compare_overview_strip.jpg` |
| **All NER→taxonomy in order** | `python -m pipeline.run_snomed_ner_chain --rf2-root "$RF2"` | Runs `ner_depth` → `graph_viz` → `pair_depths` → `branch_pairs --dedupe-concept-pairs` → `snomed_taxonomy_compare`. Needs Python **3.12**. `--skip-ner` if NER CSV already exists. Add `--graph-compact` for a hop-limited subgraph figure. |
| LLM pair review + depths (branch seeds) | `python -m pipeline.llm_pair_align --rf2-root "$RF2"` | After `branch_pairs`: Ollama → `results/llm_align/llm_pair_review.csv` + `llm_aligned_snomed_depths.csv`. |
| LLM pair review (per-case same string) | `python -m pipeline.llm_pair_align --seeds-csv results/snomed/per_case_same_string_entity_overlap.csv --max-llm-rows 48 --rf2-root "$RF2"` | Writes `llm_pair_review_per_case.csv` + `llm_aligned_snomed_depths_per_case.csv` (default paths). **No Mistral:** `python -m pipeline.llm_pair_align --depth-only --review-in results/llm_align/llm_pair_review_per_case.csv --depth-out results/llm_align/llm_aligned_snomed_depths_per_case.csv --rf2-root "$RF2"` |
| LLM pair review (per-case hierarchical seeds) | `python -m pipeline.llm_pair_align --seeds-csv results/snomed/per_case_hierarchical_entity_pairs.csv --max-llm-rows 48 --rf2-root "$RF2"` | Uses a **hierarchical-specific** rubric: prefer **KEEP** (same clinical thread → compare SNOMED depth on mapped concepts as-is; no harmonised rewrites), **INCOMPARABLE** if unrelated; **SUBSTITUTE** only for a clearly wrong automatic link. Outputs: `llm_pair_review_per_case_hierarchical.csv`, `llm_aligned_snomed_depths_per_case_hierarchical.csv`. **No Mistral:** same depth-only pattern with those paths. |
| LLM taxonomy JPEGs | `python -m pipeline.llm_taxonomy_viz --rf2-root "$RF2"` | **No Mistral:** reads `--align-csv` (default `llm_aligned_snomed_depths.csv`); add `--align-csv results/llm_align/llm_aligned_snomed_depths_per_case.csv --overview-jpeg results/llm_align/llm_taxonomy_overview_strip_per_case.jpg` for the per-case paper run. Optional `--no-category-jpeg`. |

**Resume** `prep_corpora`: skips cases where **both** output files exist and are **non-empty**. **Regenerate all:**

```bash
python -m pipeline.prep_corpora --force
```

### 40-case keyed vs.\ unkeyed ablation

Stratified indices (default quotas mirroring Table~2 area mix) and metrics JSON:

```bash
python -m pipeline.pick_unkeyed_subset --out cases_unkeyed40.txt
python -m pipeline.prep_corpora --omit-correct-option --force \
  --subset cases_unkeyed40.txt --out-corpus-b corpus_b_unkeyed/
python -m pipeline.ablation_unkeyed_stats --rf2-root "$RF2"
```

Writes `results/ablation_unkeyed40_metrics.json` (hedge and *most likely* per 1,000 tokens on the listed cases; mean SNOMED depth on triple-intersection NER entities for A, keyed B, unkeyed B).

Optional: `python -m pipeline.freq_zipf --rf2-root "..."` for the SNOMED token subset CSVs. Large branch-pair runs: add `--dedupe-concept-pairs`.

---

## Corpus B prompt (in `pipeline/prep_corpora.py`)

Ollama gets a **system** line plus a **user** message. The user block has the case (up to but not including `CORRECT ANSWER:`), states the keyed correct option, asks for a teaching-style explanation, and caps length with `num_predict` from the human word count. See the code for the exact strings.

---

## Not in this repo — AntConc

With `corpus_a/` and `corpus_b/` ready: keyword list (log-likelihood), collocations, KWIC — step-by-step notes and shell prep are in **`docs/antconc_qualitative/README.md`**.

---

## Example full sequence

**Corpus prep:** skips Mistral for any `corpus_b/case_*.txt` that already has text (unless you pass `--force`). `prep_corpora` already runs a pair-prune at the end; running `prune_corpora_pairs` again is optional but harmless.

```bash
cd "/path/to/Terminology/register_audit_mcq"
source ../.venv/bin/activate
pip install -r requirements.txt
python -m pipeline.prep_corpora
python -m pipeline.prune_corpora_pairs
python -m pipeline.freq_zipf
python -m pipeline.hedging
python -m pipeline.hedging_area
python -m pipeline.tfidf
python -m pipeline.rf2 --rf2-root "/path/to/SnomedCT_InternationalRF2_PRODUCTION_..."
python -m pipeline.ner_depth
python -m pipeline.graph_viz
python -m pipeline.pair_depths
python -m pipeline.branch_pairs
python -m pipeline.snomed_taxonomy_compare
```

### Default RF2 folder next to the project

If you unpacked SNOMED under the repo with the name in `pipeline/paths.py` (`SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z`), you can omit `--rf2-root` on `rf2`, `ner_depth`, `graph_viz`, `pair_depths`, `branch_pairs`, and `snomed_taxonomy_compare` (each module has that default). Otherwise set `RF2` once:

```bash
cd "/path/to/Terminology/register_audit_mcq"
source ../.venv/bin/activate
RF2="/path/to/SnomedCT_InternationalRF2_PRODUCTION_..."
python -m pipeline.rf2 --rf2-root "$RF2"
python -m pipeline.ner_depth --rf2-root "$RF2"
python -m pipeline.graph_viz --rf2-root "$RF2"
python -m pipeline.pair_depths
python -m pipeline.branch_pairs --rf2-root "$RF2"
python -m pipeline.snomed_taxonomy_compare --rf2-root "$RF2"
```
