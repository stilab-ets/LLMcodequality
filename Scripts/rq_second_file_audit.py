#!/usr/bin/env python3
"""
rq_second_file_audit.py

Per-agent analysis of the second file in human (gold) patches that touch 2+ files.

Uses the RAW gold patches from the SWE-Bench Pro dataset (before any filtering),
so that test and configuration files included by human developers are visible.

For each agent, among the paired succeeded instances where the human patch
touches ≥ 2 files, classify the second file path as:
  - test       : path contains test/spec markers
  - config_doc : configuration or documentation file
  - source     : other production source code

Outputs → Scripts/results/rq/
  second_file_audit.csv          — per-instance rows
  second_file_audit_summary.csv  — per-agent summary counts and percentages
  rq_table_second_file.tex       — LaTeX table
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent
SUCCEEDED = ROOT / "python_data" / "succeeded"
OUT_DIR   = Path(__file__).parent / "results" / "rq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load raw gold patches from dataset ─────────────────────────────────────────
print("Loading raw gold patches from SWE-Bench Pro dataset...")
_DATASET_URL = "hf://datasets/ScaleAI/SWE-bench_Pro/data/test-00000-of-00001.parquet"
_ds = pd.read_parquet(_DATASET_URL, columns=["instance_id", "patch"])
GOLD_PATCH: dict[str, str] = dict(zip(_ds["instance_id"], _ds["patch"]))
print(f"  → {len(GOLD_PATCH)} gold patches loaded\n")

MODELS = [
    "claude-45haiku-10222025",
    "claude-45sonnet-10132025",
    "codex",
    "gemini-2-5-pro-nov17",
    "glm-4p5-10222025",
    "gpt-5-high-paper",
    "gptoss-paper",
]

MODEL_LABELS = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}

# ── File classifier ────────────────────────────────────────────────────────────

_TEST_RE = re.compile(
    r"(?:^|/)(tests?|spec|__tests?__)(?:/|$)"
    r"|(?:^|/)conftest\.py$"
    r"|_test\.[a-z]+$"
    r"|\.test\.[a-z]+$"
    r"|\.spec\.[a-z]+$"
    r"|_spec\.[a-z]+$",
    re.IGNORECASE,
)

_CONFIG_DOC_RE = re.compile(
    r"(?:^|/)(docs?|doc)(?:/|$)"
    r"|(?:^|/)(?:README|CHANGELOG|CHANGES|CONTRIBUTING|HISTORY|AUTHORS|LICENSE|NOTICE)"
    r"|(?:setup\.(?:py|cfg))$"
    r"|(?:pyproject\.toml|tox\.ini|Makefile|Dockerfile)$"
    r"|\.(md|rst|txt|cfg|ini|toml|yaml|yml|json)$",
    re.IGNORECASE,
)


def classify(path: str) -> str:
    if _TEST_RE.search(path):
        return "test"
    if _CONFIG_DOC_RE.search(path):
        return "config_doc"
    return "source"


# ── Parse diff text ────────────────────────────────────────────────────────────

def get_files_from_text(text: str) -> list[str]:
    return re.findall(r"^diff --git a/(.+?) b/", text, re.MULTILINE)


# ── Collect per-instance data ──────────────────────────────────────────────────

records = []
for model in MODELS:
    model_dir = SUCCEEDED / model
    if not model_dir.is_dir():
        print(f"  SKIP {model} — directory not found")
        continue
    for inst_dir in sorted(model_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        iid = inst_dir.name
        raw_gold = GOLD_PATCH.get(iid, "")
        if not raw_gold:
            continue
        files = get_files_from_text(raw_gold)
        n_files = len(files)
        if n_files < 2:
            continue
        second_file = files[1]
        cat = classify(second_file)
        records.append({
            "model":        model,
            "label":        MODEL_LABELS[model],
            "instance":     iid,
            "n_gold_files": n_files,
            "first_file":   files[0],
            "second_file":  second_file,
            "category":     cat,
        })

df = pd.DataFrame(records)
print(f"Collected {len(df)} instances with 2+ gold files across {df['model'].nunique()} models\n")

df.to_csv(OUT_DIR / "second_file_audit.csv", index=False)
print(f"Per-instance data → {OUT_DIR}/second_file_audit.csv")

# ── Per-agent summary ──────────────────────────────────────────────────────────

summary_rows = []
for model in MODELS:
    mdf = df[df["model"] == model]
    if mdf.empty:
        continue
    total = len(mdf)
    n_test       = (mdf["category"] == "test").sum()
    n_config_doc = (mdf["category"] == "config_doc").sum()
    n_source     = (mdf["category"] == "source").sum()
    summary_rows.append({
        "model":       model,
        "label":       MODEL_LABELS[model],
        "n_paired":    total,
        "n_test":      n_test,
        "n_config_doc": n_config_doc,
        "n_source":    n_source,
        "pct_test":      round(100 * n_test / total, 1),
        "pct_config_doc": round(100 * n_config_doc / total, 1),
        "pct_source":  round(100 * n_source / total, 1),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "second_file_audit_summary.csv", index=False)
print(f"Summary → {OUT_DIR}/second_file_audit_summary.csv\n")

print(f"{'Agent':<10} {'N':>5}  {'Test%':>6}  {'Cfg/Doc%':>8}  {'Source%':>8}")
print("-" * 45)
for _, row in summary.iterrows():
    print(f"{row['label']:<10} {row['n_paired']:>5}  {row['pct_test']:>6.1f}  "
          f"{row['pct_config_doc']:>8.1f}  {row['pct_source']:>8.1f}")

# ── LaTeX table ────────────────────────────────────────────────────────────────

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Classification of the second file in human (gold) patches that "
    r"touch $\geq 2$ files. \emph{Test}: path matches test/spec markers; "
    r"\emph{Config/Doc}: configuration or documentation file; "
    r"\emph{Source}: other production source file.}",
    r"\label{tab:second-file-audit}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"\textbf{Agent} & \textbf{N} & \textbf{Test (\%)} & \textbf{Config/Doc (\%)} & \textbf{Source (\%)} \\",
    r"\midrule",
]

for _, row in summary.iterrows():
    lines.append(
        rf"{row['label']} & {row['n_paired']} & "
        rf"{row['pct_test']:.1f} & {row['pct_config_doc']:.1f} & {row['pct_source']:.1f} \\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

tex_out = OUT_DIR / "rq_table_second_file.tex"
tex_out.write_text("\n".join(lines))
print(f"\nLaTeX table → {tex_out}")
