#!/usr/bin/env python3
"""
rq_patch_file_types.py

For every eval instance per model, classify ALL files in both the AI patch
and the raw gold patch (before any filtering) as:
  - test       : test/spec files
  - config_doc : configuration or documentation files
  - source     : other production source code

Reports per model:
  - how many patches touch 1 / 2+ / 3+ files
  - % of all changed files that are test / config_doc / source
  - for patches with 2+ files: breakdown of the second file's category
  - side-by-side: AI vs Human

Outputs → Scripts/results/rq/
  patch_file_types_instances.csv  — per-instance detail
  patch_file_types_summary.csv    — per-model summary
  rq_table_file_types.tex         — LaTeX comparison table
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent.parent
SUCCEEDED = ROOT / "python_data" / "succeeded"
OUT_DIR   = Path(__file__).parent / "results" / "rq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Load raw gold patches ───────────────────────────────────────────────────────
print("Loading raw gold patches from SWE-Bench Pro dataset...")
_DATASET_URL = "hf://datasets/ScaleAI/SWE-bench_Pro/data/test-00000-of-00001.parquet"
_ds = pd.read_parquet(_DATASET_URL, columns=["instance_id", "patch"])
GOLD_PATCH: dict[str, str] = dict(zip(_ds["instance_id"], _ds["patch"]))
print(f"  → {len(GOLD_PATCH)} gold patches loaded\n")

# ── File classifier ─────────────────────────────────────────────────────────────

_TEST_RE = re.compile(
    r"(?:^|/)(tests?|spec|__tests?__|testdata)(?:/|$)"
    r"|(?:^|/)conftest\.py$"
    r"|_test\.[a-zA-Z0-9]+$"
    r"|\.test\.[a-zA-Z0-9]+$"
    r"|\.spec\.[a-zA-Z0-9]+$"
    r"|_spec\.[a-zA-Z0-9]+$"
    r"|(?:^|/)test_[^/]+$",
    re.IGNORECASE,
)

_CONFIG_DOC_RE = re.compile(
    r"(?:^|/)(docs?|doc|documentation)(?:/|$)"
    r"|(?:^|/)(?:README|CHANGELOG|CHANGES|CONTRIBUTING|HISTORY|AUTHORS|"
    r"LICENSE|NOTICE|INSTALL|TODO|ROADMAP)(?:\.[a-z]+)?$"
    r"|(?:setup\.(?:py|cfg))$"
    r"|(?:pyproject\.toml|tox\.ini|Makefile|Dockerfile|docker-compose\.[^/]+)$"
    r"|(?:\.github/|\.circleci/|\.travis\.yml|Jenkinsfile)"
    r"|\.(md|rst|txt|cfg|ini|toml|yaml|yml|json|xml|html|pot|po|desktop|asciidoc)$",
    re.IGNORECASE,
)


def classify(path: str) -> str:
    if _TEST_RE.search(path):
        return "test"
    if _CONFIG_DOC_RE.search(path):
        return "config_doc"
    return "source"


def get_files(text: str) -> list[str]:
    return re.findall(r"^diff --git a/(.+?) b/", text, re.MULTILINE)


def read_diff(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


# ── Collect per-instance data ───────────────────────────────────────────────────

records = []

for model in MODELS:
    model_dir = SUCCEEDED / model
    if not model_dir.is_dir():
        print(f"  SKIP {model} — directory not found")
        continue

    n_missing_gold = 0
    for inst_dir in sorted(model_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        iid = inst_dir.name

        ai_text   = read_diff(inst_dir / "_patch.diff")
        gold_text = GOLD_PATCH.get(iid, "")

        if not ai_text:
            continue
        if not gold_text:
            n_missing_gold += 1
            continue

        ai_files   = get_files(ai_text)
        gold_files = get_files(gold_text)

        if not ai_files and not gold_files:
            continue

        def summarise(files: list[str]) -> dict:
            cats = [classify(f) for f in files]
            c = Counter(cats)
            n = len(files)
            return {
                "n_files":        n,
                "n_test":         c["test"],
                "n_config_doc":   c["config_doc"],
                "n_source":       c["source"],
                "second_file":    files[1] if n >= 2 else "",
                "second_cat":     classify(files[1]) if n >= 2 else "none",
            }

        ai_s   = summarise(ai_files)
        gold_s = summarise(gold_files)

        records.append({
            "model":    model,
            "label":    MODEL_LABELS[model],
            "instance": iid,
            **{f"ai_{k}": v for k, v in ai_s.items()},
            **{f"gold_{k}": v for k, v in gold_s.items()},
        })

    n = sum(1 for r in records if r["model"] == model)
    print(f"  {MODEL_LABELS[model]:<8} {n:>4} instances  (missing gold: {n_missing_gold})")

df = pd.DataFrame(records)
print(f"\nTotal: {len(df)} instances across {df['model'].nunique()} models")
df.to_csv(OUT_DIR / "patch_file_types_instances.csv", index=False)
print(f"Per-instance data → {OUT_DIR}/patch_file_types_instances.csv\n")

# ── Per-model summary ───────────────────────────────────────────────────────────

def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


summary_rows = []
for model in MODELS:
    mdf = df[df["model"] == model]
    if mdf.empty:
        continue

    for side, prefix in [("AI", "ai"), ("Human", "gold")]:
        total_patches = len(mdf)
        total_files   = mdf[f"{prefix}_n_files"].sum()
        n_single      = (mdf[f"{prefix}_n_files"] == 1).sum()
        n_multi       = (mdf[f"{prefix}_n_files"] >= 2).sum()
        n_three_plus  = (mdf[f"{prefix}_n_files"] >= 3).sum()

        n_test        = mdf[f"{prefix}_n_test"].sum()
        n_config_doc  = mdf[f"{prefix}_n_config_doc"].sum()
        n_source      = mdf[f"{prefix}_n_source"].sum()

        # second-file category among multi-file patches
        multi = mdf[mdf[f"{prefix}_n_files"] >= 2]
        n2 = len(multi)
        second_test    = (multi[f"{prefix}_second_cat"] == "test").sum()
        second_cfg_doc = (multi[f"{prefix}_second_cat"] == "config_doc").sum()
        second_source  = (multi[f"{prefix}_second_cat"] == "source").sum()

        summary_rows.append({
            "model":         model,
            "label":         MODEL_LABELS[model],
            "side":          side,
            "n_patches":     total_patches,
            "n_single_file": n_single,
            "n_multi_file":  n_multi,
            "n_three_plus":  n_three_plus,
            "pct_single":    pct(n_single, total_patches),
            "pct_multi":     pct(n_multi, total_patches),
            "total_files":   int(total_files),
            "pct_file_test": pct(n_test, total_files),
            "pct_file_cfg":  pct(n_config_doc, total_files),
            "pct_file_src":  pct(n_source, total_files),
            "n_multi_pairs": n2,
            "pct_2nd_test":  pct(second_test, n2),
            "pct_2nd_cfg":   pct(second_cfg_doc, n2),
            "pct_2nd_src":   pct(second_source, n2),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "patch_file_types_summary.csv", index=False)
print(f"Summary → {OUT_DIR}/patch_file_types_summary.csv\n")

# ── Console output ──────────────────────────────────────────────────────────────

print("─── File-type breakdown across ALL changed files ─────────────────────────")
print(f"{'Agent':<8} {'Side':<6} {'Patches':>7}  {'1-file%':>7}  {'2+file%':>7}  "
      f"{'Test%':>6}  {'Cfg/Doc%':>8}  {'Source%':>7}")
print("─" * 75)
for _, row in summary.iterrows():
    print(f"{row['label']:<8} {row['side']:<6} {row['n_patches']:>7}  "
          f"{row['pct_single']:>7.1f}  {row['pct_multi']:>7.1f}  "
          f"{row['pct_file_test']:>6.1f}  {row['pct_file_cfg']:>8.1f}  "
          f"{row['pct_file_src']:>7.1f}")
    if row['side'] == 'Human':
        print()

print("\n─── Second-file category (among 2+ file patches) ─────────────────────────")
print(f"{'Agent':<8} {'Side':<6} {'N(2+)':>6}  {'2nd=Test%':>9}  {'2nd=Cfg%':>8}  {'2nd=Src%':>8}")
print("─" * 58)
for _, row in summary.iterrows():
    print(f"{row['label']:<8} {row['side']:<6} {row['n_multi_pairs']:>6}  "
          f"{row['pct_2nd_test']:>9.1f}  {row['pct_2nd_cfg']:>8.1f}  "
          f"{row['pct_2nd_src']:>8.1f}")
    if row['side'] == 'Human':
        print()

# ── LaTeX table ─────────────────────────────────────────────────────────────────

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{File-type breakdown of AI and human patches (before filtering). "
    r"\emph{1-file\%}: patches touching a single file; \emph{2+file\%}: patches touching "
    r"two or more files. File categories: \emph{Test} = test/spec files; "
    r"\emph{Cfg/Doc} = configuration or documentation; \emph{Source} = production source code. "
    r"``2nd file'' columns show the category of the second file among multi-file patches.}",
    r"\label{tab:file-types}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{llrrrrrrrrrr}",
    r"\toprule",
    r"\textbf{Agent} & \textbf{Patch} & \textbf{N} & \textbf{1-file\%} & \textbf{2+file\%} "
    r"& \textbf{All-Test\%} & \textbf{All-Cfg\%} & \textbf{All-Src\%} "
    r"& \textbf{2nd=Test\%} & \textbf{2nd=Cfg\%} & \textbf{2nd=Src\%} \\",
    r"\midrule",
]

for model in MODELS:
    msub = summary[summary["model"] == model]
    if msub.empty:
        continue
    for i, (_, row) in enumerate(msub.iterrows()):
        label = row["label"] if i == 0 else ""
        lines.append(
            rf"{label} & {row['side']} & {row['n_patches']} "
            rf"& {row['pct_single']:.1f} & {row['pct_multi']:.1f} "
            rf"& {row['pct_file_test']:.1f} & {row['pct_file_cfg']:.1f} & {row['pct_file_src']:.1f} "
            rf"& {row['pct_2nd_test']:.1f} & {row['pct_2nd_cfg']:.1f} & {row['pct_2nd_src']:.1f} \\"
        )
    lines.append(r"\midrule")

lines[-1] = r"\bottomrule"
lines += [
    r"\end{tabular}}",
    r"\end{table*}",
]

tex_out = OUT_DIR / "rq_table_file_types.tex"
tex_out.write_text("\n".join(lines))
print(f"\nLaTeX table → {tex_out}")
