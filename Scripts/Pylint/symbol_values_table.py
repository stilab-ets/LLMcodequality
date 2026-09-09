#!/usr/bin/env python3
"""
symbol_values_table.py

Read symbol_values_llm.csv / symbol_values_gold.csv per instance, compute
per-instance mean diff (after - before), then compare LLM vs Gold with
Wilcoxon + Cliff's delta.  Outputs a LaTeX table and box plots.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

PYLINT_ROOT = Path("pylint_results")
OUT_DIR     = PYLINT_ROOT / "results"
BOX_DIR     = OUT_DIR / "boxplots_symbol_values"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOX_DIR.mkdir(parents=True, exist_ok=True)

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

GOLD_COLOR = "#f4a261"
LLM_COLOR  = "#4c72b0"

SYMBOL_GROUPS = [
    ("Size", [
        ("line-too-long",       "Line Length"),
        ("too-many-lines",      "Module Lines"),
        ("too-many-statements", "Statements"),
        ("too-many-locals",     "Local Vars"),
    ]),
    ("Complexity", [
        ("too-many-branches",            "Branches"),
        ("too-many-nested-blocks",       "Nested Blocks"),
        ("too-many-boolean-expressions", "Bool Expressions"),
        ("too-many-return-statements",   "Return Stmts"),
        ("too-many-try-statements",      "Try Stmts"),
    ]),
    ("OOP Design", [
        ("too-many-arguments",           "Arguments"),
        ("too-many-instance-attributes", "Instance Attrs"),
        ("too-many-public-methods",      "Public Methods"),
        ("too-few-public-methods",       "Few Pub. Methods"),
    ]),
]

MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_test(gold: pd.Series, llm: pd.Series,
             gold_central: pd.Series, llm_central: pd.Series):
    """Wilcoxon on (gold, llm); central tendency from the non-zero subsets."""
    pairs = pd.DataFrame({"g": gold, "l": llm}).dropna()
    if len(pairs) < 10:
        return np.nan, np.nan, np.nan, np.nan, "N"
    d   = pairs["l"] - pairs["g"]
    h_c = gold_central.dropna().median() if not gold_central.dropna().empty else np.nan
    l_c = llm_central.dropna().median()  if not llm_central.dropna().empty  else np.nan
    if (d == 0).all():
        return 1.0, h_c, l_c, 0.0, "N"
    try:
        _, p = wilcoxon(pairs["l"], pairs["g"], zero_method="pratt", alternative="two-sided")
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, "N"
    cd, mag = cliffs_delta_lib(pairs["l"].tolist(), pairs["g"].tolist())
    return p, h_c, l_c, cd, MAG_SHORT.get(mag, mag[0].upper())


def fmt_p(p) -> str:
    if pd.isna(p):  return r"\text{--}"
    if p < 0.001:   return r"\textbf{***}"
    if p < 0.01:    return r"\textbf{**}"
    if p < 0.05:    return r"\textbf{*}"
    return f"{p:.3f}"


def fmt_val(v) -> str:
    return r"\text{--}" if pd.isna(v) else f"{v:.2f}"


# ── Load per-instance mean diffs from symbol_values_{variant}.csv ─────────────

def load_diffs(model_dir: Path, variant: str) -> pd.DataFrame:
    records = []
    for path in model_dir.rglob(f"symbol_values_{variant}.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "symbol" not in df.columns:
            continue
        df["before_value"] = pd.to_numeric(df["before_value"], errors="coerce")
        df["after_value"]  = pd.to_numeric(df["after_value"],  errors="coerce")

        # keep only rows where both before and after exist
        matched = df.dropna(subset=["before_value", "after_value"]).copy()
        if matched.empty:
            continue

        matched["diff"] = matched["after_value"] - matched["before_value"]
        for sym, grp in matched.groupby("symbol"):
            records.append({"instance_id": path.parent.name, "symbol": sym,
                            "diff": grp["diff"].mean()})
    return pd.DataFrame(records)


# ── Load all models ────────────────────────────────────────────────────────────

print("Loading data ...")
data_by_model = {}
for model in MODELS:
    d = PYLINT_ROOT / model
    if not d.exists():
        continue
    llm  = load_diffs(d, "llm").rename(columns={"diff": "diff_llm"})
    gold = load_diffs(d, "diff_gold" if False else "gold").rename(columns={"diff": "diff_gold"})
    merged = pd.merge(llm, gold, on=["instance_id", "symbol"], how="outer")
    if merged.empty:
        continue
    data_by_model[model] = merged
    print(f"  {model}: {merged['instance_id'].nunique()} instances, {merged['symbol'].nunique()} symbols")

# ── Compute stats ──────────────────────────────────────────────────────────────

stats_by_model = {}
for model, merged in data_by_model.items():
    all_instances = set(merged["instance_id"].unique())
    rows = []
    for group, symbols in SYMBOL_GROUPS:
        for sym, display_name in symbols:
            sub = merged[merged["symbol"] == sym]
            # non-zero subset for central tendency
            g_nz = sub["diff_gold"].dropna()
            l_nz = sub["diff_llm"].dropna()
            # zero-filled for statistical test (no change = 0)
            full = (sub.set_index("instance_id")
                       .reindex(all_instances)
                       .fillna(0)
                       .reset_index())
            p, h_c, l_c, cd, mag = run_test(
                full["diff_gold"], full["diff_llm"], g_nz, l_nz)
            rows.append({"group": group, "symbol": sym, "display_name": display_name,
                         "gold_central": h_c, "model_central": l_c,
                         "p": p, "cliffs_delta": cd, "delta_mag": mag})
    stats_by_model[model] = pd.DataFrame(rows)
    stats_by_model[model].to_csv(OUT_DIR / f"symbol_values_stats_{model}.csv", index=False)

# ── Box plots ──────────────────────────────────────────────────────────────────

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

print(f"\nGenerating box plots → {BOX_DIR}/")
for group, symbols in SYMBOL_GROUPS:
    for sym, display_name in symbols:
        model_data = []
        for model, merged in data_by_model.items():
            sub = merged[merged["symbol"] == sym]
            if sub.empty:
                continue
            model_data.append((MODEL_DISPLAY.get(model, model),
                                sub["diff_gold"].dropna().values,
                                sub["diff_llm"].dropna().values))
        if not model_data:
            continue

        n = len(model_data)
        fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 2.6), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, (label, g_vals, l_vals) in zip(axes, model_data):
            for vals, color, pos in [(g_vals, GOLD_COLOR, -0.25), (l_vals, LLM_COLOR, 0.25)]:
                bp = ax.boxplot(vals if len(vals) else [0], positions=[pos], widths=0.5,
                                patch_artist=True, notch=False, showfliers=True,
                                medianprops=dict(color="black", linewidth=1.5),
                                flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                                                markerfacecolor=color, alpha=0.5))
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.9)
            all_v = list(g_vals[np.isfinite(g_vals)]) + list(l_vals[np.isfinite(l_vals)])
            if all_v:
                v_min, v_max = min(all_v), max(all_v)
                pad = (v_max - v_min) * 0.15 or 0.5
                ax.set_ylim(v_min - pad, v_max + pad)
            ax.set_xticks([-0.25, 0.25])
            ax.set_xticklabels(["G", "L"], fontsize=7)
            ax.set_title(label, fontsize=8, pad=3)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(display_name)
        fig.legend(handles=[mpatches.Patch(facecolor=GOLD_COLOR, label="Gold"),
                             mpatches.Patch(facecolor=LLM_COLOR,  label="LLM")],
                   loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(BOX_DIR / f"{group}_{sym}.pdf".replace(" ", "_")), bbox_inches="tight")
        plt.close(fig)

print("  Done.")

# ── LaTeX table ────────────────────────────────────────────────────────────────

print("\nGenerating LaTeX table ...")
model_names = list(stats_by_model.keys())
n_models    = len(model_names)

header = " & ".join(
    rf"\multicolumn{{5}}{{c}}{{\textbf{{{MODEL_DISPLAY.get(m, m)}}}}}" for m in model_names)
subheader = " & ".join(
    r"\textbf{Gold} & \textbf{LLM} & \textbf{$p$} & \textbf{$\delta$} & \textbf{Mag.}"
    for _ in model_names)
col_spec = "ll " + " ".join(["r r r r c"] * n_models)
cmidrules = " ".join(rf"\cmidrule(lr){{{3+i*5}-{7+i*5}}}" for i in range(n_models))

lines = [
    r"\begin{table*}[t]", r"\centering",
    r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.1}",
    r"\caption{All models: LLM vs Gold patch Pylint threshold values — "
    r"median of per-instance mean diff (after$-$before), "
    r"Wilcoxon $p$-value, Cliff's $\delta$. "
    r"$p$: *** $<$ 0.001, ** $<$ 0.01, * $<$ 0.05. Mag.: N/S/M/L.}",
    r"\label{tab:pylint_symbol_values}",
    r"\resizebox{0.9\textwidth}{!}{",
    rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
    rf"\textbf{{Quality Attr.}} & \textbf{{Symbol}} & {header} \\",
    cmidrules,
    rf"\textbf{{}} & \textbf{{}} & {subheader} \\",
    r"\midrule",
]

end_col = 2 + n_models * 5
for group, symbols in SYMBOL_GROUPS:
    lines.append(r"\midrule")
    n_rows = len(symbols)
    for i, (sym, display_name) in enumerate(symbols):
        group_cell = rf"\multirow{{{n_rows}}}{{*}}{{{group}}}" if i == 0 else ""
        sym_tex    = rf"\texttt{{{sym.replace('-', r'\text{-}')}}}"
        cols = []
        for m in model_names:
            df = stats_by_model.get(m)
            row = df[df["symbol"] == sym] if df is not None else pd.DataFrame()
            if row.empty:
                cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
            else:
                r  = row.iloc[0]
                cd = f"{r['cliffs_delta']:.3f}" if not pd.isna(r["cliffs_delta"]) else r"\text{--}"
                cols.append(f"{fmt_val(r['gold_central'])} & {fmt_val(r['model_central'])} & "
                            f"{fmt_p(r['p'])} & {cd} & ({r['delta_mag']})")
        lines.append(f"{group_cell} & {sym_tex} & " + " & ".join(cols) + r" \\")
        if i < n_rows - 1:
            lines.append(rf"\cmidrule(lr){{2-{end_col}}}")

lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]

tex_path = OUT_DIR / "symbol_values_table_all_models.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  LaTeX table → {tex_path.name}")
print("\nDone.")
