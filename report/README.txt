LaTeX must be run from THIS directory (report/), because main.tex uses
\graphicspath{{../results/}} relative to the shell's current working directory.

  cd "/path/to/Terminology/register_audit_mcq/report"
  pdflatex -interaction=nonstopmode main
  pdflatex -interaction=nonstopmode main

The document uses a manual \texttt{thebibliography} block in \texttt{main.tex}; there is no \texttt{bibtex} step. Layout matches the parallel \texttt{clinical\_text\_simplification/report} project: spanning title/abstract, a small table of contents, then two-column body, \texttt{table*}/\texttt{figure*} for wide floats, \texttt{[t]} placement. SNOMED detail lives in Appendix~A; branch-pair grid in Appendix~B.

Before \texttt{pdflatex}, run from the repo root: \texttt{python3 report/gen\_report\_figures.py} (TF--IDF and LLM bar PNGs plus \texttt{report/\_generated\_llm\_hier.tex} counts).

Use "main" not "main.tex" if you like; both work once you are in report/.

Do NOT run pdflatex from the Terminology parent folder: it will fail with
"I can't find file main.tex" unless you pass the full path—and even then,
figures may break unless the CWD is report/.

Figures load from ../results/ (run the full pipeline first). LLM taxonomy
JPEGs: from register_audit_mcq/ run  python -m pipeline.llm_taxonomy_viz
so ../results/llm_align/*.jpg exists before compiling.

Optional graph-bundled judge JSON (Ollama): from register_audit_mcq/ run
  python -m pipeline.model_explanation_graph_rag_judge case_005.txt
which writes ../results/llm_align/model_explanation_graph_judge_005.json (see ../docs/command_prompts.md).

Proposal checklist and AntConc steps: ../docs/proposal_alignment.md
