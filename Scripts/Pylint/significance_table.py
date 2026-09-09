#!/usr/bin/env python3
"""
significance_table.py

For each model and each Pylint category (convention, refactor, warning, error, fatal, total),
compare the LLM patch diff vs the Gold (human) patch diff using:
  - Wilcoxon signed-rank test  (significance)
  - Cliff's delta              (effect-size magnitude)

Output (written to pylint_results/results/):
  - significance_stats_<model>.csv   — per-model raw stats
  - significance_table_all_models.tex — combined LaTeX table

Flags:
  --central mean|median       Central tendency for the table (default: median)
  --remove-outliers           Clip values outside [0.5%, 99.5%] before analysis
  --outlier-threshold FLOAT   Percentile threshold (default: 0.5)
  --results-dir PATH          Root of pylint_results/ (default: pylint_results)
  --categories CATS           Space-separated list of categories to include
                              (default: convention refactor warning error fatal total)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--central", choices=["mean", "median"], default="median")
parser.add_argument("--remove-outliers", action="store_true", default=False)
parser.add_argument("--outlier-threshold", type=float, default=0.5)
parser.add_argument("--results-dir", default="pylint_results")
parser.add_argument(
    "--categories",
    nargs="+",
    default=["convention", "refactor", "warning", "error", "fatal", "total"],
    metavar="CAT",
)
args = parser.parse_args()

USE_MEDIAN      = args.central == "median"
REMOVE_OUTLIERS = args.remove_outliers
OUTLIER_PCT     = args.outlier_threshold / 100.0
CATEGORIES      = args.categories

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path(args.results_dir)
OUT_DIR      = RESULTS_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-codex-debug-oct22":  "GPT-5D",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}

MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}

# ── Helpers ────────────────────────────────────────────────────────────────────


def clip_outliers(series: pd.Series) -> pd.Series:
    if not REMOVE_OUTLIERS:
        return series
    lo = series.quantile(OUTLIER_PCT)
    hi = series.quantile(1.0 - OUTLIER_PCT)
    return series.where((series >= lo) & (series <= hi))


def run_test(human_vals: pd.Series, llm_vals: pd.Series):
    """Wilcoxon signed-rank on paired diffs; returns (p, gold_central, llm_central, cd, mag)."""
    pairs = pd.DataFrame({"h": human_vals, "l": llm_vals}).dropna()
    if len(pairs) < 10:
        return np.nan, np.nan, np.nan, np.nan, "N"

    agg = pairs.median if USE_MEDIAN else pairs.mean
    d   = pairs["l"] - pairs["h"]

    if (d == 0).all():
        return 1.0, agg()["h"], agg()["l"], 0.0, "N"

    try:
        _, p = wilcoxon(pairs["l"], pairs["h"], zero_method="pratt", alternative="two-sided")
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, "N"

    cd, mag = cliffs_delta_lib(pairs["l"].tolist(), pairs["h"].tolist())
    return p, agg()["h"], agg()["l"], cd, MAG_SHORT.get(mag, mag[0].upper())


def fmt_p(p) -> str:
    if pd.isna(p):  return r"\text{--}"
    if p < 0.001:   return r"\textbf{***}"
    if p < 0.01:    return r"\textbf{**}"
    if p < 0.05:    return r"\textbf{*}"
    return f"{p:.3f}"


def fmt_val(v) -> str:
    if pd.isna(v): return r"\text{--}"
    return f"{v:.2f}"


# ── Discover models ────────────────────────────────────────────────────────────

MODELS: dict[str, Path] = {}
for d in sorted(RESULTS_ROOT.iterdir()):
    if not d.is_dir():
        continue
    csv = d / "final_aggregated_metrics.csv"
    if csv.exists():
        MODELS[d.name] = csv

if not MODELS:
    raise FileNotFoundError(f"No final_aggregated_metrics.csv found under {RESULTS_ROOT}")

print(f"Found {len(MODELS)} model(s): {list(MODELS)}\n")

# ── Compute stats per model ────────────────────────────────────────────────────

stats_by_model: dict[str, pd.DataFrame] = {}

for model_name, csv_path in MODELS.items():
    df = pd.read_csv(csv_path)
    print(f"  {model_name}: {len(df)} instances")

    rows = []
    for cat in CATEGORIES:
        h_col = f"diff_n_{cat}_human"
        l_col = f"diff_n_{cat}_llm"
        if h_col not in df.columns or l_col not in df.columns:
            print(f"    SKIP {cat} — column missing")
            continue

        p, h_c, l_c, cd, dmag = run_test(
            clip_outliers(pd.to_numeric(df[h_col], errors="coerce")),
            clip_outliers(pd.to_numeric(df[l_col], errors="coerce")),
        )
        rows.append({
            "category":      cat,
            "gold_central":  h_c,
            "model_central": l_c,
            "p":             p,
            "cliffs_delta":  cd,
            "delta_mag":     dmag,
        })

    df_res = pd.DataFrame(rows)
    stats_by_model[model_name] = df_res

    out_csv = OUT_DIR / f"significance_stats_{model_name}.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"    Stats CSV → {out_csv.name}")

# ── LaTeX table ────────────────────────────────────────────────────────────────

print("\nGenerating LaTeX table ...")

central_label = "Median" if USE_MEDIAN else "Mean"
model_names   = list(stats_by_model.keys())
n_models      = len(model_names)

model_header_parts = " & ".join(
    rf"\multicolumn{{5}}{{c}}{{\textbf{{{MODEL_DISPLAY.get(m, m)}}}}}"
    for m in model_names
)
sub_parts = " & ".join(
    r"\textbf{Gold} & \textbf{LLM} & \textbf{$p$} & \textbf{$\delta$} & \textbf{Mag.}"
    for _ in model_names
)
col_spec = "l " + " ".join(["r r r r c"] * n_models)

lines: list[str] = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\setlength{\tabcolsep}{3pt}")
lines.append(r"\renewcommand{\arraystretch}{1.1}")
lines.append(
    rf"\caption{{Pylint category deltas — {central_label}, "
    r"Wilcoxon $p$-value, and Cliff's $\delta$ effect size "
    r"(LLM patch vs Gold patch). "
    r"$p$: *** $<$ 0.001, ** $<$ 0.01, * $<$ 0.05. "
    r"Mag.: N=negligible, S=small, M=medium, L=large.}}"
)
lines.append(r"\label{tab:pylint_significance}")
lines.append(r"\resizebox{\columnwidth}{!}{")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
lines.append(rf"\textbf{{Category}} & {model_header_parts} \\")

cmidrules = " ".join(
    rf"\cmidrule(lr){{{2 + i*5}-{6 + i*5}}}" for i in range(n_models)
)
lines.append(cmidrules)
lines.append(rf"\textbf{{}} & {sub_parts} \\")
lines.append(r"\midrule")

for cat in CATEGORIES:
    model_cols = []
    for model_name in model_names:
        df_res = stats_by_model.get(model_name)
        if df_res is None:
            model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
            continue
        row = df_res[df_res["category"] == cat]
        if row.empty:
            model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
        else:
            r = row.iloc[0]
            cd_str = f"{r['cliffs_delta']:.3f}" if not pd.isna(r["cliffs_delta"]) else r"\text{--}"
            model_cols.append(
                f"{fmt_val(r['gold_central'])} & {fmt_val(r['model_central'])} & "
                f"{fmt_p(r['p'])} & {cd_str} & ({r['delta_mag']})"
            )

    lines.append(rf"\textbf{{{cat.capitalize()}}} & " + " & ".join(model_cols) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")
lines.append(r"\end{table}")

tex_path = OUT_DIR / "significance_table_all_models.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  LaTeX table → {tex_path}")

print("\nDone.")
