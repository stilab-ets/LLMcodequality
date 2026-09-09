#!/usr/bin/env python3
"""
rq_patch_diff.py

RQ: How are agent-written patches different from human (gold) ones?

Loads data from three pipelines across all 8 models:
  - Understand  : code metrics (complexity, coupling, size, …)
  - Pylint      : static-analysis issue counts
  - DPY         : code smells (arch / design / implementation)

For every (model, metric) pair it computes:
  - median gold delta   — change introduced by the human patch
  - median LLM delta    — change introduced by the agent patch
  - Wilcoxon signed-rank p-value (paired, two-sided)
  - Cliff's delta effect size + magnitude label

Outputs  (written to results/rq/):
  rq_patch_diff_summary.csv   — full results table
  rq_patch_diff.tex            — LaTeX significance table
  rq_patch_diff_heatmap.pdf   — direction × significance heatmap
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
OUT_DIR = SCRIPTS / "results" / "rq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "claude-45haiku-10222025",
    "claude-45sonnet-10132025",
    "codex",
    "gemini-2-5-pro-nov17",
    "glm-4p5-10222025",
    "gpt-5-codex-debug-oct22",
    "gpt-5-high-paper",
    "gptoss-paper",
]

MODEL_LABELS = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-codex-debug-oct22":  "GPT-5D",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}

UND_ROOT    = SCRIPTS / "Understand"  / "understand_results"
PYL_ROOT    = SCRIPTS / "Pylint"      / "pylint_results"
DPY_AGG     = SCRIPTS / "DPY"         / "dpy_results" / "aggregated.csv"

# ── Metric definitions ─────────────────────────────────────────────────────────
# Understand: (display, category, raw_metric, level)
UND_METRICS = [
    # Complexity
    ("CC",       "Complexity",   "Cyclomatic",          "function"),
    ("WMC",      "Complexity",   "SumCyclomatic",       "file"),
    ("MaxNest",  "Complexity",   "MaxNesting",          "function"),
    ("NPATH",    "Complexity",   "CountPathLog",        "function"),
    # Coupling
    ("CBO",      "Coupling",     "CountClassCoupled",   "class"),
    ("RFC",      "Coupling",     "CountDeclMethodAll",  "class"),
    # Inheritance
    ("DIT",      "Inheritance",  "MaxInheritanceTree",  "class"),
    ("NOC",      "Inheritance",  "CountClassDerived",   "class"),
    ("IFANIN",   "Inheritance",  "CountClassBase",      "class"),
    # Cohesion
    ("LCOM",     "Cohesion",     "PercentLackOfCohesion", "class"),
    # Design size
    ("LOC",      "Size",         "CountLineCode",       "file"),
    ("NL",       "Size",         "CountLine",           "file"),
    ("CLOC",     "Size",         "CountLineComment",    "file"),
    ("NOM",      "Size",         "CountDeclMethod",     "class"),
    ("NIM",      "Size",         "CountDeclInstanceMethod", "class"),
    ("NIV",      "Size",         "CountDeclInstanceVariable", "class"),
]

# Pylint: (display, category, suffix)  — pairs diff_{suffix}_human / _llm
PYL_METRICS = [
    ("Issues (total)",      "Pylint",  "n_total"),
    ("Convention",          "Pylint",  "n_convention"),
    ("Refactor",            "Pylint",  "n_refactor"),
    ("Warning",             "Pylint",  "n_warning"),
    ("Error",               "Pylint",  "n_error"),
]

# DPY smells: (display, category, prefix)
# net change = {prefix}_added - {prefix}_removed  for gold_ and llm_
DPY_SMELLS = [
    # Architectural
    ("God Component",           "Arch Smell",   "arch_god_component"),
    ("Unstable Dependency",     "Arch Smell",   "arch_unstable_dependency"),
    # Design
    ("Broken Hierarchy",        "Design Smell", "design_broken_hierarchy"),
    ("Broken Modularization",   "Design Smell", "design_broken_modularization"),
    ("Deficient Encapsulation", "Design Smell", "design_deficient_encapsulation"),
    ("Feature Envy",            "Design Smell", "design_feature_envy"),
    ("Insuf. Modularization",   "Design Smell", "design_insufficient_modularization"),
    ("Wide Hierarchy",          "Design Smell", "design_wide_hierarchy"),
    # Implementation
    ("Complex Method",          "Impl Smell",   "impl_complex_method"),
    ("Long Method",             "Impl Smell",   "impl_long_method"),
    ("Long Parameter List",     "Impl Smell",   "impl_long_parameter_list"),
    ("Magic Number",            "Impl Smell",   "impl_magic_number"),
    ("Long Statement",          "Impl Smell",   "impl_long_statement"),
    ("Complex Conditional",     "Impl Smell",   "impl_complex_conditional"),
]

# ── Statistical helpers ────────────────────────────────────────────────────────
MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}


def run_test(gold: pd.Series, llm: pd.Series):
    pairs = pd.DataFrame({"g": gold, "l": llm}).dropna()
    if len(pairs) < 10:
        return dict(n=len(pairs), gold_med=np.nan, llm_med=np.nan,
                    p=np.nan, cd=np.nan, mag="N", sig=False)
    d = pairs["l"] - pairs["g"]
    gold_med = pairs["g"].median()
    llm_med  = pairs["l"].median()
    if (d == 0).all():
        return dict(n=len(pairs), gold_med=gold_med, llm_med=llm_med,
                    p=1.0, cd=0.0, mag="N", sig=False)
    try:
        _, p = wilcoxon(pairs["l"], pairs["g"], zero_method="pratt",
                        alternative="two-sided")
    except Exception:
        p = np.nan
    cd, mag = cliffs_delta_lib(pairs["l"].values.tolist(), pairs["g"].values.tolist())
    return dict(n=len(pairs), gold_med=gold_med, llm_med=llm_med,
                p=p, cd=cd, mag=MAG_SHORT.get(mag, mag[0].upper()),
                sig=(not np.isnan(p)) and p < 0.05)


# ── Collect results ────────────────────────────────────────────────────────────
rows = []

# DPY (single aggregated file, model column present)
if DPY_AGG.exists():
    dpy_df = pd.read_csv(DPY_AGG)
    for model in MODELS:
        mdf = dpy_df[dpy_df["model"] == model]
        if mdf.empty:
            continue
        for display, category, prefix in DPY_SMELLS:
            g_add = f"gold_{prefix}_added"
            g_rem = f"gold_{prefix}_removed"
            l_add = f"llm_{prefix}_added"
            l_rem = f"llm_{prefix}_removed"
            if not all(c in mdf.columns for c in [g_add, g_rem, l_add, l_rem]):
                continue
            gold_net = mdf[g_add].fillna(0) - mdf[g_rem].fillna(0)
            llm_net  = mdf[l_add].fillna(0) - mdf[l_rem].fillna(0)
            stat = run_test(gold_net, llm_net)
            rows.append({"source": "DPY", "category": category,
                         "metric": display, "model": model,
                         "model_label": MODEL_LABELS[model], **stat})
else:
    print(f"  SKIP DPY — {DPY_AGG} not found")

# Understand + Pylint (per-model files)
for model in MODELS:
    und_csv = UND_ROOT / model / "final_aggregated_metrics.csv"
    pyl_csv = PYL_ROOT / model / "final_aggregated_metrics.csv"

    if und_csv.exists():
        df = pd.read_csv(und_csv)
        for display, category, raw, level in UND_METRICS:
            h_col = f"diff_{raw}_{level}_human"
            l_col = f"diff_{raw}_{level}_llm"
            if h_col not in df.columns or l_col not in df.columns:
                continue
            stat = run_test(df[h_col], df[l_col])
            rows.append({"source": "Understand", "category": category,
                         "metric": display, "model": model,
                         "model_label": MODEL_LABELS[model], **stat})
    else:
        print(f"  SKIP Understand/{model}")

    if pyl_csv.exists():
        df = pd.read_csv(pyl_csv)
        for display, category, suffix in PYL_METRICS:
            h_col = f"diff_{suffix}_human"
            l_col = f"diff_{suffix}_llm"
            if h_col not in df.columns or l_col not in df.columns:
                continue
            stat = run_test(df[h_col], df[l_col])
            rows.append({"source": "Pylint", "category": category,
                         "metric": display, "model": model,
                         "model_label": MODEL_LABELS[model], **stat})
    else:
        print(f"  SKIP Pylint/{model}")

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "rq_patch_diff_summary.csv", index=False)
print(f"Summary saved → {OUT_DIR}/rq_patch_diff_summary.csv  ({len(summary)} rows)")

# ── Heatmap: direction × significance ─────────────────────────────────────────
# One heatmap per source, rows = metrics, cols = models
# Cell value = Cliff's delta (signed by direction: LLM>Gold positive)
# Significance overlay: hatching on non-significant cells

MODEL_ORDER  = [MODEL_LABELS[m] for m in MODELS if m in summary["model"].values]

SOURCES = [
    ("Understand", "Understand metrics"),
    ("Pylint",     "Pylint issues"),
    ("DPY",        "Code smells (DPY)"),
]

for src, src_title in SOURCES:
    sdf = summary[summary["source"] == src].copy()
    if sdf.empty:
        continue

    metric_order = (
        sdf.groupby(["category", "metric"])
        .size().reset_index()[["category", "metric"]]
        .drop_duplicates()
    )
    metrics = list(metric_order["metric"])
    models  = [l for l in MODEL_ORDER if l in sdf["model_label"].values]

    # Build signed-cd matrix and significance mask
    cd_mat  = pd.DataFrame(np.nan,  index=metrics, columns=models)
    sig_mat = pd.DataFrame(False,   index=metrics, columns=models)

    for _, row in sdf.iterrows():
        m, ml = row["metric"], row["model_label"]
        if m in cd_mat.index and ml in cd_mat.columns:
            cd_mat.loc[m, ml]  = row["cd"] if not np.isnan(row.get("cd", np.nan)) else np.nan
            sig_mat.loc[m, ml] = bool(row["sig"])

    fig, ax = plt.subplots(figsize=(max(4, len(models) * 0.7 + 1.5), max(3, len(metrics) * 0.38 + 1)))

    vmax = max(0.6, np.nanmax(np.abs(cd_mat.values)))
    im = ax.imshow(cd_mat.values.astype(float), aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Hatch non-significant cells
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            val = cd_mat.loc[metric, model]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color="lightgrey", zorder=2))
            elif not sig_mat.loc[metric, model]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, hatch="////",
                                           edgecolor="white", linewidth=0, zorder=2))

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=7)
    ax.set_title(f"{src_title}\n(color = Cliff's δ direction; hatch = not significant p≥0.05)",
                 fontsize=8, pad=6)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Cliff's δ  (blue = LLM < Gold,  red = LLM > Gold)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fname = OUT_DIR / f"rq_heatmap_{src.lower()}.pdf"
    fig.savefig(str(fname), bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {fname}")

# ── LaTeX table: significant results only ─────────────────────────────────────
sig_df = summary[summary["sig"]].copy()
sig_df["direction"] = sig_df.apply(
    lambda r: r"$\uparrow$" if r["llm_med"] > r["gold_med"] else r"$\downarrow$", axis=1)

def fmt(v):
    return r"\text{--}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return r"\text{--}"
    if p < 0.001: return r"\textbf{<.001}"
    if p < 0.01:  return r"\textbf{<.01}"
    if p < 0.05:  return r"\textbf{<.05}"
    return f"{p:.3f}"

lines = [
    r"\begin{longtable}{llllrrrll}",
    r"\toprule",
    r"\textbf{Source} & \textbf{Category} & \textbf{Metric} & \textbf{Model} "
    r"& \textbf{$\tilde{x}$ Gold} & \textbf{$\tilde{x}$ LLM} "
    r"& \textbf{$p$} & \textbf{$\delta$} & \textbf{Dir.} \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endfoot",
]

prev_src = prev_cat = None
for _, r in sig_df.sort_values(["source", "category", "metric", "model_label"]).iterrows():
    src_cell = r["source"]  if r["source"]   != prev_src else ""
    cat_cell = r["category"] if r["category"] != prev_cat else ""
    prev_src = r["source"]; prev_cat = r["category"]
    lines.append(
        f"{src_cell} & {cat_cell} & {r['metric']} & {r['model_label']} "
        f"& {fmt(r['gold_med'])} & {fmt(r['llm_med'])} "
        f"& {fmt_p(r['p'])} & {fmt(r['cd'])} & {r['direction']} \\\\"
    )

lines.append(r"\end{longtable}")

tex_path = OUT_DIR / "rq_patch_diff.tex"
tex_path.write_text("\n".join(lines))
print(f"LaTeX table saved → {tex_path}  ({len(sig_df)} significant rows)")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────────")
total   = len(summary)
sig_cnt = summary["sig"].sum()
print(f"Total (model, metric) pairs tested : {total}")
print(f"Significant (p < 0.05)             : {sig_cnt}  ({100*sig_cnt/total:.1f}%)")
print()

for src in ["Understand", "Pylint", "DPY"]:
    sub = summary[summary["source"] == src]
    s   = sub[sub["sig"]]
    if sub.empty: continue
    up   = (s["llm_med"] > s["gold_med"]).sum()
    down = (s["llm_med"] < s["gold_med"]).sum()
    print(f"{src:12s}  tested={len(sub):3d}  sig={len(s):3d}  "
          f"LLM↑={up}  LLM↓={down}")

print()
print("Top significant differences (by |Cliff's δ|):")
top = (summary[summary["sig"]]
       .assign(abs_cd=summary["cd"].abs())
       .sort_values("abs_cd", ascending=False)
       .head(15))
for _, r in top.iterrows():
    direction = "LLM > Gold" if r["llm_med"] > r["gold_med"] else "LLM < Gold"
    print(f"  [{r['source']:10s}] {r['metric']:28s} {r['model_label']:8s} "
          f"δ={r['cd']:+.2f} ({r['mag']})  p={r['p']:.3f}  {direction}")
