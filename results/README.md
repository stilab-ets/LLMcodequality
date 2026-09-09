# `results/` — analysis output, organized by research question

Statistics CSVs and box plots that back the paper, grouped by the research
question they answer. The pipelines under `Scripts/` write their raw per-model /
per-instance output into their own `*_results/` trees (git-ignored); the curated
copy lives here. LaTeX table files (`*.tex`) are regenerated from these CSVs and
are git-ignored.

| RQ | Question | Tools |
|----|----------|-------|
| **RQ1** | How do LLM-generated patches differ structurally from human patches? | diff-level analysis (`Scripts/rq_*.py`) |
| **RQ2** | How does the structural *quality* compare? (7 attributes / 26 metrics) | SciTools Understand + DesignitePy |
| **RQ3** | How do they differ in adherence to coding standards? | Pylint |

