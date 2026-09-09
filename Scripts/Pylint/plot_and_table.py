#!/usr/bin/env python3
"""
plot_and_table.py

Compare each model's LLM patch vs its own Gold (human) patch for the
10 most frequent Pylint symbols per message type (convention, error,
refactor, warning, fatal).

For each instance and symbol, counts are tracked separately:
  - introduced: count(status == "new")
  - removed:    count(status == "removed")

The table has TWO rows per symbol — one for introduced, one for removed.

Flags:
  --central mean|median      Central tendency for the table (default: median)
  --remove-outliers          Clip values outside [0.5%, 99.5%] before all analysis
  --outlier-threshold FLOAT  Percentile threshold for clipping (default: 0.5)
  --top-n INT                Number of top symbols per type to include (default: 10)

Outputs (written to pylint_results/results/):
  - One PDF box plot per symbol (all models, gold vs llm, introduced & removed)
    → results/boxplots/
  - Combined LaTeX table (all models, each llm vs its own gold, 2 rows per symbol)
    → results/comparison_table_all_models.tex
  - Per-model stats CSV
    → results/comparison_stats_<model>.csv
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--central", choices=["mean", "median"], default="median")
parser.add_argument("--remove-outliers", action="store_true", default=False)
parser.add_argument("--outlier-threshold", type=float, default=0.5)
parser.add_argument("--top-n", type=int, default=10,
                    help="Number of top symbols for non-convention types (default: 10)")
parser.add_argument("--top-n-convention", type=int, default=20,
                    help="Number of top symbols for convention type (default: 20)")
args = parser.parse_args()

USE_MEDIAN        = args.central == "median"
REMOVE_OUTLIERS   = args.remove_outliers
OUTLIER_PCT       = args.outlier_threshold / 100.0
TOP_N             = args.top_n
TOP_N_CONVENTION  = args.top_n_convention

# ── Paths ──────────────────────────────────────────────────────────────────────
PYLINT_ROOT = Path("/home/AV00500/Issam/SWE-Bench_Pro/swebench_pro_pipeline/Scripts/Pylint/pylint_results")
OUT_DIR     = PYLINT_ROOT / "results"
BOX_DIR     = OUT_DIR / "boxplots"
OUT_DIR.mkdir(exist_ok=True)
BOX_DIR.mkdir(exist_ok=True)

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

GOLD_COLOR = "#f4a261"
LLM_COLOR  = "#4c72b0"

MODEL_COLORS = {
    "claude-45haiku-10222025":  "#1f77b4",
    "claude-45sonnet-10132025": "#ff7f0e",
    "codex":                    "#2ca02c",
    "gemini-2-5-pro-nov17":     "#d62728",
    "glm-4p5-10222025":         "#9467bd",
    "gpt-5-codex-debug-oct22":  "#8c564b",
    "gpt-5-high-paper":         "#e377c2",
    "gptoss-paper":             "#7f7f7f",
}

MODEL_DISPLAY_NAMES = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-codex-debug-oct22":  "GPT-5D",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}

# Message types to include (fatal has only one symbol so still include)
MSG_TYPES = ["convention", "error", "refactor", "warning", "fatal"]

# ── Statistical helpers ────────────────────────────────────────────────────────

MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}


def remove_outliers_series(series: pd.Series) -> pd.Series:
    if not REMOVE_OUTLIERS:
        return series
    lo = series.quantile(OUTLIER_PCT)
    hi = series.quantile(1.0 - OUTLIER_PCT)
    return series.where((series >= lo) & (series <= hi))


def run_test(human_vals, llm_vals):
    pairs = pd.DataFrame({"h": human_vals, "l": llm_vals}).dropna()
    if len(pairs) < 10:
        return np.nan, np.nan, np.nan, np.nan, "N"
    d = pairs["l"] - pairs["h"]
    agg = lambda s: s.median() if USE_MEDIAN else s.mean()
    if (d == 0).all():
        return 1.0, agg(pairs["h"]), agg(pairs["l"]), 0.0, "N"
    try:
        _, p = wilcoxon(pairs["l"], pairs["h"], zero_method="pratt", alternative="two-sided")
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, "N"
    cd, mag = cliffs_delta_lib(pairs["l"].values.tolist(), pairs["h"].values.tolist())
    return p, agg(pairs["h"]), agg(pairs["l"]), cd, MAG_SHORT.get(mag, mag[0].upper())


def fmt_p(p):
    if np.isnan(p): return r"\text{--}"
    if p < 0.001:   return r"\textbf{***}"
    if p < 0.01:    return r"\textbf{**}"
    if p < 0.05:    return r"\textbf{*}"
    return f"{p:.3f}"


def fmt_val(v):
    if np.isnan(v): return r"\text{--}"
    return f"{v:.2f}"


# ── Load per-instance symbol counts from messages_diff_*.csv ──────────────────
# Returns DataFrame columns: instance_id, symbol, type,
#   new_llm, removed_llm, new_gold, removed_gold

def compute_symbol_counts(model_name: str) -> pd.DataFrame:
    """
    For each instance of *model_name*, collect per-symbol counts of
    introduced (status=="new") and removed (status=="removed") messages
    for both llm and gold variants.

    Netting: for each (symbol, obj) pair within an instance, introduced
    and removed occurrences cancel each other out — only the surplus in
    either direction is counted.
    """
    model_dir = PYLINT_ROOT / model_name
    if not model_dir.exists():
        return pd.DataFrame()

    # instance -> symbol -> obj -> {type, new, removed}
    llm_rows  = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"type": None, "new": 0, "removed": 0})))
    gold_rows = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"type": None, "new": 0, "removed": 0})))

    for variant, store in (("llm", llm_rows), ("gold", gold_rows)):
        for csv_path in model_dir.rglob(f"messages_diff_{variant}.csv"):
            instance_id = csv_path.parent.name
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty or "symbol" not in df.columns:
                continue
            for _, row in df.iterrows():
                sym      = row["symbol"]
                msg_type = row.get("type", None)
                status   = row.get("status", None)
                obj      = row.get("obj", "__no_obj__")
                rec = store[instance_id][sym][obj]
                if rec["type"] is None and msg_type:
                    rec["type"] = msg_type
                if status == "new":
                    rec["new"] += 1
                elif status == "removed":
                    rec["removed"] += 1

    def _net_counts(sym_obj_dict):
        """Sum netted (introduced, removed) across all objects for one (instance, symbol)."""
        sym_type = None
        net_new = net_rem = 0
        for rec in sym_obj_dict.values():
            if sym_type is None and rec["type"]:
                sym_type = rec["type"]
            surplus = rec["new"] - rec["removed"]
            if surplus > 0:
                net_new += surplus
            elif surplus < 0:
                net_rem += abs(surplus)
        return sym_type, net_new, net_rem

    records = []
    all_instances = set(llm_rows.keys()) | set(gold_rows.keys())
    for inst in all_instances:
        all_syms = set(llm_rows[inst].keys()) | set(gold_rows[inst].keys())
        for sym in all_syms:
            l_type, l_new, l_rem = _net_counts(llm_rows[inst].get(sym, {}))
            g_type, g_new, g_rem = _net_counts(gold_rows[inst].get(sym, {}))
            records.append({
                "instance_id":  inst,
                "symbol":       sym,
                "type":         l_type or g_type,
                "new_llm":      l_new,
                "removed_llm":  l_rem,
                "new_gold":     g_new,
                "removed_gold": g_rem,
            })

    return pd.DataFrame(records)


# ── Identify top-N symbols per type across all models ─────────────────────────

print("Loading data and computing symbol counts ...")

data_by_model: dict[str, pd.DataFrame] = {}

for model in MODELS:
    df = compute_symbol_counts(model)
    if df.empty:
        print(f"  SKIP {model} — no data found")
        continue
    data_by_model[model] = df
    print(f"  {model}: {df['instance_id'].nunique()} instances, "
          f"{df['symbol'].nunique()} symbols")

# Rank symbols by cross-model occurrence count (rows = instance-symbol pairs)
all_combined = pd.concat(data_by_model.values(), ignore_index=True)

top_symbols_by_type: dict[str, list[str]] = {}
for msg_type, grp in all_combined.groupby("type"):
    if msg_type not in MSG_TYPES:
        continue
    n = TOP_N_CONVENTION if msg_type == "convention" else TOP_N
    counts = grp["symbol"].value_counts()
    top_symbols_by_type[msg_type] = counts.head(n).index.tolist()
    print(f"  Top {n} [{msg_type}]: {top_symbols_by_type[msg_type]}")


# ── Compute per-model stats (introduced + removed separately) ─────────────────

stats_by_model: dict[str, pd.DataFrame] = {}

for model_name, df in data_by_model.items():
    rows = []
    for msg_type, symbols in top_symbols_by_type.items():
        for sym in symbols:
            sub = df[df["symbol"] == sym]
            if sub.empty:
                continue
            for direction, h_vals, l_vals in [
                ("introduced",
                 remove_outliers_series(pd.to_numeric(sub["new_gold"],     errors="coerce")),
                 remove_outliers_series(pd.to_numeric(sub["new_llm"],      errors="coerce"))),
                ("removed",
                 remove_outliers_series(pd.to_numeric(sub["removed_gold"], errors="coerce")),
                 remove_outliers_series(pd.to_numeric(sub["removed_llm"],  errors="coerce"))),
                ("net impact",
                 remove_outliers_series(pd.to_numeric(sub["new_gold"], errors="coerce")
                                        - pd.to_numeric(sub["removed_gold"], errors="coerce")),
                 remove_outliers_series(pd.to_numeric(sub["new_llm"],  errors="coerce")
                                        - pd.to_numeric(sub["removed_llm"],  errors="coerce"))),
            ]:
                p, h_c, l_c, cd, dmag = run_test(h_vals, l_vals)
                rows.append({
                    "type":          msg_type,
                    "symbol":        sym,
                    "direction":     direction,
                    "gold_central":  h_c,
                    "model_central": l_c,
                    "p":             p,
                    "cliffs_delta":  cd,
                    "delta_mag":     dmag,
                })

    df_res = pd.DataFrame(rows)
    stats_by_model[model_name] = df_res
    df_res.to_csv(OUT_DIR / f"comparison_stats_{model_name}.csv", index=False)
    print(f"  {model_name}: stats CSV saved.")


# ── Box plots ──────────────────────────────────────────────────────────────────

print(f"\nGenerating box plots → {BOX_DIR}/")

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

box_w     =  0.5
pos_human = -box_w / 2
pos_llm   =  box_w / 2

for msg_type, symbols in top_symbols_by_type.items():
    for sym in symbols:
        model_data = []
        for model_name, df in data_by_model.items():
            sub = df[df["symbol"] == sym]
            if sub.empty:
                continue
            model_data.append((
                model_name,
                remove_outliers_series(pd.to_numeric(sub["new_gold"],     errors="coerce")).dropna().values,
                remove_outliers_series(pd.to_numeric(sub["new_llm"],      errors="coerce")).dropna().values,
                remove_outliers_series(pd.to_numeric(sub["removed_gold"], errors="coerce")).dropna().values,
                remove_outliers_series(pd.to_numeric(sub["removed_llm"],  errors="coerce")).dropna().values,
            ))

        if not model_data:
            continue

        n = len(model_data)
        fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 5.2), sharey=False)
        if n == 1:
            axes = axes.reshape(2, 1)

        gold_patch  = mpatches.Patch(facecolor=GOLD_COLOR, alpha=0.9, label="Gold")
        model_patch = mpatches.Patch(facecolor=LLM_COLOR,  alpha=0.9, label="LLM")

        for col_i, (model_name, new_g, new_l, rem_g, rem_l) in enumerate(model_data):
            for row_i, (g_vals, l_vals, row_label) in enumerate([
                (new_g, new_l, "Introduced"),
                (rem_g, rem_l, "Removed"),
            ]):
                ax = axes[row_i][col_i]
                for vals, color, pos in [(g_vals, GOLD_COLOR, pos_human),
                                         (l_vals, LLM_COLOR,  pos_llm)]:
                    bp = ax.boxplot(vals if len(vals) else [0],
                                    positions=[pos], widths=box_w,
                                    patch_artist=True, notch=False, showfliers=True,
                                    medianprops=dict(color="black", linewidth=1.5),
                                    whiskerprops=dict(linewidth=1.0),
                                    capprops=dict(linewidth=1.0),
                                    boxprops=dict(linewidth=1.0),
                                    flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                                                    markerfacecolor=color, alpha=0.5))
                    bp["boxes"][0].set_facecolor(color)
                    bp["boxes"][0].set_alpha(0.9)

                all_vals = list(g_vals[np.isfinite(g_vals)]) + list(l_vals[np.isfinite(l_vals)])
                if all_vals:
                    v_min, v_max = min(all_vals), max(all_vals)
                    pad = (v_max - v_min) * 0.15 or 0.5
                    ax.set_ylim(v_min - pad, v_max + pad)

                ax.set_xticks([pos_human, pos_llm])
                ax.set_xticklabels(["G", "L"], fontsize=7)
                ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                ax.set_axisbelow(True)
                if row_i == 0:
                    ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name), fontsize=8, pad=3)
                if col_i == 0:
                    ax.set_ylabel(row_label, fontsize=8)

        fig.suptitle(f"[{msg_type}] {sym}", fontsize=8, y=1.02)
        fig.legend(handles=[gold_patch, model_patch],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=2, frameon=False, fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        safe = f"{msg_type}_{sym}".replace("-", "_").replace(" ", "_")
        fig.savefig(str(BOX_DIR / f"{safe}.pdf"), bbox_inches="tight")
        plt.close(fig)

print("  Done.")


# ── Combined LaTeX table ────────────────────────────────────────────────────────

print("\nGenerating combined LaTeX table ...")

central_label = "median" if USE_MEDIAN else "mean"
model_names   = list(stats_by_model.keys())
n_models      = len(model_names)

# 3 label columns: Type | Symbol | Direction
model_header_parts = " & ".join(
    rf"\multicolumn{{5}}{{c}}{{\textbf{{{MODEL_DISPLAY_NAMES.get(m, m)}}}}}"
    for m in model_names
)
sub_parts = " & ".join(
    r"\textbf{Gold} & \textbf{LLM} & \textbf{$p$} & \textbf{$\delta$} & \textbf{Mag.}"
    for _ in model_names
)
col_spec = "lll " + " ".join(["r r r r c"] * n_models)

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\setlength{\tabcolsep}{3pt}")
lines.append(r"\renewcommand{\arraystretch}{1.1}")
lines.append(
    rf"\caption{{All models: LLM vs Gold patch Pylint symbol counts — "
    rf"top {TOP_N} symbols per type, {central_label} per instance, "
    rf"Wilcoxon $p$-value, Cliff's $\delta$.}}"
)
lines.append(r"\label{tab:pylint_all_models_llm_vs_gold}")
lines.append(r"\resizebox{0.9\textwidth}{!}{")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
lines.append(rf"\textbf{{Type}} & \textbf{{Symbol}} & \textbf{{Direction}} & {model_header_parts} \\")
cmidrules = " ".join(
    rf"\cmidrule(lr){{{4 + i*5}-{8 + i*5}}}" for i in range(n_models)
)
lines.append(cmidrules)
lines.append(rf"\textbf{{}} & \textbf{{}} & \textbf{{}} & {sub_parts} \\")
lines.append(r"\midrule")

DIRECTIONS = ["introduced", "removed", "net impact"]
DIR_LABELS = {
    "introduced": r"\textit{intro.}",
    "removed":    r"\textit{rem.}",
    "net impact": r"\textit{net}",
}

for msg_type in MSG_TYPES:
    if msg_type not in top_symbols_by_type:
        continue
    symbols = top_symbols_by_type[msg_type]
    n_type_rows = len(symbols) * 3

    lines.append(r"\midrule")

    for sym_idx, sym in enumerate(symbols):
        sym_display = sym.replace("_", r"\_").replace("-", r"\text{-}")

        for dir_idx, direction in enumerate(DIRECTIONS):
            type_cell = (rf"\multirow{{{n_type_rows}}}{{*}}{{\textit{{{msg_type}}}}}"
                         if sym_idx == 0 and dir_idx == 0 else "")
            sym_cell  = (rf"\multirow{{3}}{{*}}{{\texttt{{{sym_display}}}}}"
                         if dir_idx == 0 else "")

            model_cols = []
            for model_name in model_names:
                df_res = stats_by_model.get(model_name)
                if df_res is None:
                    model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
                    continue
                row = df_res[
                    (df_res["type"] == msg_type) &
                    (df_res["symbol"] == sym) &
                    (df_res["direction"] == direction)
                ]
                if row.empty:
                    model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
                else:
                    r  = row.iloc[0]
                    cd = r["cliffs_delta"]
                    cd_str = f"{cd:.3f}" if not np.isnan(cd) else r"\text{--}"
                    model_cols.append(
                        f"{fmt_val(r['gold_central'])} & {fmt_val(r['model_central'])} & "
                        f"{fmt_p(r['p'])} & {cd_str} & ({r['delta_mag']})"
                    )

            lines.append(
                f"{type_cell} & {sym_cell} & {DIR_LABELS[direction]} & "
                + " & ".join(model_cols) + r" \\"
            )

            if dir_idx < 2:
                end_col = 3 + n_models * 5
                lines.append(rf"\cmidrule(lr){{3-{end_col}}}")

        if sym_idx < len(symbols) - 1:
            end_col = 3 + n_models * 5
            lines.append(rf"\cmidrule(lr){{2-{end_col}}}")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")
lines.append(r"\end{table*}")

tex_path = OUT_DIR / "comparison_table_all_models.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  LaTeX table → {tex_path.name}")

print("\nDone.")
