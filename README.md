# Replication package — LLM-agent vs. human patch quality on SWE-Bench Pro

This repository is the replication package for an empirical study that compares
**LLM-agent patches** against **matched human (gold) patches** on the
[SWE-Bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) benchmark.

For every issue an agent solved, the agent patch and the human patch start from
the **same base commit** and both pass the **same tests**. Three static-analysis
pipelines — **SciTools Understand**, **DesignitePy**, and **Pylint** — measure the
before→after delta of each patch, and the two deltas are compared pairwise with a
**paired two-sided Wilcoxon signed-rank test**.

## Research questions

| RQ  | Question | Instruments |
|-----|----------|-------------|
| **RQ1** | How do LLM-generated patches differ *structurally* from human patches? | diff-level analysis (`Scripts/rq_*.py`) |
| **RQ2** | How does the structural *quality* compare? (7 quality attributes / 26 metrics) | Understand + DesignitePy |
| **RQ3** | How do they differ in *adherence to coding standards*? | Pylint |

## Scope

- **Languages / repos:** Python only — `ansible__ansible`, `qutebrowser__qutebrowser`,
  `internetarchive__openlibrary` (the 3 Python repos of SWE-Bench Pro's 11).
- **Models (7 in the paper):** `claude-45haiku-10222025`, `claude-45sonnet-10132025`,
  `codex`, `gemini-2-5-pro-nov17`, `glm-4p5-10222025`, `gpt-5-high-paper`,
  `gptoss-paper`. An 8th model, `gpt-5-codex-debug-oct22`, was collected and still
  appears in the raw per-model data but is **excluded from the paper**.
- **Instance naming:** `instance_{owner}__{repo}-{commit_hash}-v{version_hash}`
  (`commit_hash` = commit being fixed, `version_hash` = eval environment version).

## Repository layout

```
.
├── Scripts/               analysis pipelines + cross-pipeline RQ scripts
│   ├── Understand/        SciTools Understand metrics (CLI `und`)
│   ├── DPY/               DesignitePy code-smell analysis
│   ├── Pylint/            Pylint coding-standard diagnostics
│   ├── rq_*.py            cross-pipeline, paper-facing analyses
├── results/              all statistics CSVs + figures, organized by RQ
│   ├── RQ1/  RQ2/  RQ3/  combined/
│   └── README.md         per-file provenance (which script produced what)
├── python_data/
│   └── succeeded/{model}/{instance}/_patch.diff   the agent patches (study input)
└── README.md
```
## Reproducing the analysis

Each pipeline shares the same module layout (`config.py`, `dataset_loader.py`,
`repo_manager.py`, `patch_cleaner.py`, `pipeline_runner.py`, `diff_computer.py`,
`metrics_aggregator.py`, `plot_and_table.py`) and runs **one model at a time** via
`MODEL` in `config.py` (or `--model`). Checked-out repositories are expected at
`repos/{model}/{repo}/`.

| Pipeline | External tool | Entry point |
|----------|---------------|-------------|
| `Scripts/Understand/` | SciTools Understand `und` CLI | `python main.py` (set `MODEL`) |
| `Scripts/DPY/` | `DPy` (DesignitePy) | `python main.py`; `python run_all_models.py` for the batch |
| `Scripts/Pylint/` | `pylint` | `python main.py` (set `MODEL`) |
