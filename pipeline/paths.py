# folders and output files for this project
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
CORPUS_A = ROOT / "corpus_a"
CORPUS_B = ROOT / "corpus_b"
RESULTS = ROOT / "results"

DIR_FREQ = RESULTS / "corpus_frequency"
DIR_HEDGING = RESULTS / "hedging"
DIR_HEDGING_SPECIALTY = DIR_HEDGING / "specialty_lexicon_tagging"
DIR_TFIDF = RESULTS / "tfidf"
DIR_SNOMED = RESULTS / "snomed"
DIR_TAXONOMY_JPEG = DIR_SNOMED / "taxonomy_compare_by_category_jpeg"
DIR_LLM_ALIGN = RESULTS / "llm_align"
LLM_PAIR_REVIEW_CSV = DIR_LLM_ALIGN / "llm_pair_review.csv"
LLM_ALIGNED_DEPTHS_CSV = DIR_LLM_ALIGN / "llm_aligned_snomed_depths.csv"
LLM_PAIR_REVIEW_PER_CASE = DIR_LLM_ALIGN / "llm_pair_review_per_case.csv"
LLM_PAIR_REVIEW_PER_CASE_HIERARCHICAL = DIR_LLM_ALIGN / "llm_pair_review_per_case_hierarchical.csv"
LLM_ALIGNED_DEPTHS_PER_CASE = DIR_LLM_ALIGN / "llm_aligned_snomed_depths_per_case.csv"
LLM_ALIGNED_DEPTHS_PER_CASE_HIERARCHICAL = (
    DIR_LLM_ALIGN / "llm_aligned_snomed_depths_per_case_hierarchical.csv"
)
DIR_LLM_TAXONOMY_JPEG = DIR_LLM_ALIGN / "taxonomy_jpeg"
LLM_TAXONOMY_OVERVIEW = DIR_LLM_ALIGN / "llm_taxonomy_overview_strip.jpg"
LLM_TAXONOMY_OVERVIEW_PER_CASE = DIR_LLM_ALIGN / "llm_taxonomy_overview_strip_per_case.jpg"

FREQ_TOP_HUMAN = DIR_FREQ / "human_top50_token_freq.csv"
FREQ_TOP_MODEL = DIR_FREQ / "model_top50_token_freq.csv"
FREQ_SNOMED_HUMAN = DIR_FREQ / "human_top50_freq_snomed_token_subset.csv"
FREQ_SNOMED_MODEL = DIR_FREQ / "model_top50_freq_snomed_token_subset.csv"
ZIPF_PLOT = DIR_FREQ / "zipf_rank_frequency_loglog.png"

HEDGE_SUMMARY = DIR_HEDGING / "hedge_density_both_corpora.csv"
HEDGE_CASE_TABLE = DIR_HEDGING_SPECIALTY / "per_case_specialty_and_hedge_rates.csv"
HEDGE_GROUP_TABLE = DIR_HEDGING_SPECIALTY / "mean_hedge_by_specialty_group.csv"
HEDGE_PLOT_OVERALL = DIR_HEDGING_SPECIALTY / "bar_overall_human_vs_model.png"
HEDGE_PLOT_BY_SPECIALTY = DIR_HEDGING_SPECIALTY / "bar_by_specialty_group.png"

TFIDF_GAP = DIR_TFIDF / "shared_terms_mean_tfidf_gap.csv"

SNOMED_DEPTH_FREQ = DIR_SNOMED / "snomed_depth_from_freq_top_terms.csv"
SNOMED_DEPTH_NER = DIR_SNOMED / "snomed_depth_from_ner_entities.csv"
SNOMED_SUBGRAPH = DIR_SNOMED / "plot_subgraph_high_depth_ner.png"
SNOMED_PAIR_DEPTH_CSV = DIR_SNOMED / "paired_entity_depth_same_string.csv"
SNOMED_PAIR_HIST = DIR_SNOMED / "hist_depth_difference_b_minus_a.png"
SNOMED_PER_CASE_SAME_STRING = DIR_SNOMED / "per_case_same_string_entity_overlap.csv"
SNOMED_PER_CASE_HIERARCHICAL = DIR_SNOMED / "per_case_hierarchical_entity_pairs.csv"
SNOMED_PER_CASE_AB_SUMMARY = DIR_SNOMED / "per_case_ab_snomed_summary.csv"
SNOMED_PER_CASE_AB_INVENTORY = DIR_SNOMED / "per_case_ab_snomed_entity_inventory.csv"
SNOMED_PER_CASE_AB_CROSS = DIR_SNOMED / "per_case_ab_snomed_cross_concepts.csv"
SNOMED_PER_CASE_AB_DASHBOARD = DIR_SNOMED / "per_case_ab_snomed_dashboard.png"
SNOMED_PER_CASE_AB_CROSS_FIG = DIR_SNOMED / "per_case_ab_snomed_cross_pairs.png"
SNOMED_BRANCH_PAIRS = DIR_SNOMED / "branch_pairs_all_concepts.csv"
SNOMED_BRANCH_CATEGORIES = DIR_SNOMED / "branch_pairs_category_counts.csv"
SNOMED_TAXONOMY_OVERVIEW = DIR_SNOMED / "taxonomy_compare_overview_strip.jpg"

HF_CACHE = ROOT / ".hf_cache"
NLTK_DATA = ROOT / ".nltk_data"

RF2_RELEASE_DIR = "SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z"
DEFAULT_RF2_ROOT = REPO_ROOT / RF2_RELEASE_DIR


def explain_missing_ner_csv(ner_path: Path | None = None) -> str:
    import sys

    path = ner_path or SNOMED_DEPTH_NER
    return (
        f"Missing NER output file:\n  {path}\n\n"
        "Create it first (spaCy; use Python 3.12 — 3.14 is not supported):\n"
        f"  {sys.executable} -m pipeline.ner_depth --rf2-root <PATH_TO_RF2>\n\n"
        "Or run the whole NER → taxonomy chain in one go:\n"
        f"  {sys.executable} -m pipeline.run_snomed_ner_chain --rf2-root <PATH_TO_RF2>\n"
    )


def ensure_results() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_FREQ.mkdir(parents=True, exist_ok=True)
    DIR_HEDGING.mkdir(parents=True, exist_ok=True)
    DIR_HEDGING_SPECIALTY.mkdir(parents=True, exist_ok=True)
    DIR_TFIDF.mkdir(parents=True, exist_ok=True)
    DIR_SNOMED.mkdir(parents=True, exist_ok=True)
    DIR_TAXONOMY_JPEG.mkdir(parents=True, exist_ok=True)
    DIR_LLM_ALIGN.mkdir(parents=True, exist_ok=True)
    DIR_LLM_TAXONOMY_JPEG.mkdir(parents=True, exist_ok=True)
