#!/usr/bin/env python3
"""
plot_and_table.py

Compare each model's LLM patch vs its own Gold (human) patch metric diffs per instance.
Each model's data comes from a single final_aggregated_metrics.csv that contains both
diff_<Metric>_<level>_human and diff_<Metric>_<level>_llm columns.

All models are shown together in combined plots and a single combined table.

Flags:
  --central mean|median      Central tendency for the table (default: median)
  --remove-outliers          Clip values outside [0.5%, 99.5%] before all analysis
  --outlier-threshold FLOAT  Percentile threshold for clipping (default: 0.5)

Outputs (written to understand_results/results/):
  - One PDF box plot per metric (all models, gold vs llm side-by-side) → results/boxplots/
  - Combined LaTeX table (all models, each llm vs its own gold)        → results/comparison_table_all_models.tex
  - Per-model stats CSV                                                → results/comparison_stats_<model>.csv

Data from:
  Scripts/Understand/understand_results/<model>/final_aggregated_metrics.csv
"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--central", choices=["mean", "median"], default="median",
                    help="Central tendency to report in the table (default: median)")
parser.add_argument("--remove-outliers", action="store_true", default=False,
                    help="Remove per-column outliers outside the [0.5%%, 99.5%%] percentile range")
parser.add_argument("--outlier-threshold", type=float, default=0.5,
                    help="Percentile threshold for outlier removal (default: 0.5)")
parser.add_argument("--show-fliers", action="store_true", default=False,
                    help="Show outlier points in box plots (default: hidden)")
args = parser.parse_args()
USE_MEDIAN       = args.central == "median"
REMOVE_OUTLIERS  = args.remove_outliers
OUTLIER_PCT      = args.outlier_threshold / 100.0   # e.g. 0.5 → 0.005
SHOW_FLIERS      = args.show_fliers

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_RESULTS_ROOT = Path("/home/AV00500/Issam/SWE-Bench_Pro/swebench_pro_pipeline/Scripts/Understand/understand_results")
OUT_DIR            = MODEL_RESULTS_ROOT / "results"
BOX_DIR            = OUT_DIR / "boxplots"
OUT_DIR.mkdir(exist_ok=True)
BOX_DIR.mkdir(exist_ok=True)

MODELS = {
    "claude-45haiku-10222025":  MODEL_RESULTS_ROOT / "claude-45haiku-10222025"  / "final_aggregated_metrics.csv",
    "claude-45sonnet-10132025": MODEL_RESULTS_ROOT / "claude-45sonnet-10132025" / "final_aggregated_metrics.csv",
    "codex":                    MODEL_RESULTS_ROOT / "codex"                    / "final_aggregated_metrics.csv",
    "gemini-2-5-pro-nov17":     MODEL_RESULTS_ROOT / "gemini-2-5-pro-nov17"     / "final_aggregated_metrics.csv",
    "glm-4p5-10222025":         MODEL_RESULTS_ROOT / "glm-4p5-10222025"         / "final_aggregated_metrics.csv",
    "gpt-5-high-paper":         MODEL_RESULTS_ROOT / "gpt-5-high-paper"         / "final_aggregated_metrics.csv",
    "gptoss-paper":             MODEL_RESULTS_ROOT / "gptoss-paper"             / "final_aggregated_metrics.csv",
}

GOLD_COLOR = "#f4a261"
LLM_COLOR  = "#4c72b0"

MODEL_COLORS = {
    "claude-45haiku-10222025":  "#1f77b4",
    "claude-45sonnet-10132025": "#ff7f0e",
    "codex":                    "#2ca02c",
    "gemini-2-5-pro-nov17":     "#d62728",
    "glm-4p5-10222025":         "#9467bd",
    "gpt-5-high-paper":         "#e377c2",
    "gptoss-paper":             "#7f7f7f",
}

MODEL_DISPLAY_NAMES = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}

# ── Metrics map: (display_name, level, raw_metric_name) ───────────────────────
# Columns in final_aggregated_metrics.csv:
#   diff_<raw_metric_name>_<level>_human  (gold patch delta)
#   diff_<raw_metric_name>_<level>_llm    (LLM patch delta)
METRICS_MAP = {
    "Complexity": [
        ("CC",                  "function", "Cyclomatic"),
        ("WMC",                 "file",     "SumCyclomatic"),
        ("MaxNest",             "function", "MaxNesting"),
        ("NPATH",               "function", "CountPathLog"),

    ],
    "Coupling": [
        ("CBO",                 "class",    "CountClassCoupled"),
        ("RFC",                 "class",    "CountDeclMethodAll"),
    ],
    "Inheritance": [
        ("NOC",                 "class",    "CountClassDerived"),
        ("IFANIN",              "class",    "CountClassBase"),
    ],
    "Abstraction": [
        ("Declarative Methods", "class",    "CountDeclMethod"),
    ],
    "Design Size": [
        ("NL",                  "file",     "CountLine"),
        ("LOC",                 "file",     "CountLineCode"),
        ("BLOC",                "file",     "CountLineBlank"),
        ("CLOC",                "file",     "CountLineComment"),
        ("STMTC",               "file",     "CountStmt"),
        ("Classes",             "file",     "CountDeclClass"),
        ("Functions",           "file",     "CountDeclFunction"),
        ("NIM",           "class",     "CountDeclInstanceMethod"),
        ("NIV",           "class",     "CountDeclInstanceVariable"),
    ],
}

# ── Statistical helpers ────────────────────────────────────────────────────────

MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}


def remove_outliers(series: pd.Series) -> pd.Series:
    """Replace values outside [OUTLIER_PCT, 1-OUTLIER_PCT] quantile range with NaN."""
    if not REMOVE_OUTLIERS:
        return series
    lo = series.quantile(OUTLIER_PCT)
    hi = series.quantile(1.0 - OUTLIER_PCT)
    return series.where((series >= lo) & (series <= hi))


def run_test(human_vals, llm_vals):
    """Wilcoxon signed-rank on paired (human, llm) diffs after dropping NaNs."""
    pairs = pd.DataFrame({"h": human_vals, "l": llm_vals}).dropna()
    if len(pairs) < 10:
        return np.nan, np.nan, np.nan, 0.0, "N"
    d = pairs["l"] - pairs["h"]
    agg = lambda s: s.median() if USE_MEDIAN else s.mean()
    if (d == 0).all():
        return 1.0, agg(pairs["h"]), agg(pairs["l"]), 0.0, "N"
    try:
        _, p = wilcoxon(pairs["l"], pairs["h"], zero_method="pratt", alternative="two-sided")
    except Exception:
        return np.nan, np.nan, np.nan, 0.0, "N"
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


MAG_COLORS = {"S": "magnitudeS", "M": "magnitudeM", "L": "magnitudeL"}

def fmt_mag(mag):
    cell = f"({mag})"
    if mag in MAG_COLORS:
        return rf"\cellcolor{{{MAG_COLORS[mag]}}}{cell}"
    return cell


# ── Load data ──────────────────────────────────────────────────────────────────

data_by_model:  dict[str, pd.DataFrame] = {}
stats_by_model: dict[str, pd.DataFrame] = {}

for model_name, csv_path in MODELS.items():
    if not csv_path.exists():
        print(f"  SKIP {model_name} — {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    data_by_model[model_name] = df
    print(f"  {model_name}: {len(df)} instances")

    rows = []
    for attr, metrics in METRICS_MAP.items():
        for display_name, level, raw_metric in metrics:
            h_col = f"diff_{raw_metric}_{level}_human"
            l_col = f"diff_{raw_metric}_{level}_llm"
            if h_col not in df.columns or l_col not in df.columns:
                continue
            p, h_c, l_c, cd, dmag = run_test(
                remove_outliers(df[h_col]),
                remove_outliers(df[l_col]),
            )
            rows.append({
                "attr": attr, "metric": display_name,
                "gold_central": h_c, "model_central": l_c,
                "p": p, "cliffs_delta": cd, "delta_mag": dmag,
            })

    df_res = pd.DataFrame(rows)
    stats_by_model[model_name] = df_res

    df_res.to_csv(OUT_DIR / f"comparison_stats_{model_name}.csv", index=False)
    print(f"    Stats CSV saved.")


# ── Combined box plots ─────────────────────────────────────────────────────────

print(f"\nGenerating combined box plots → {BOX_DIR}/")

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  14,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

BOX_WIDTH  = 0.15
GROUP_GAP  = 0.15
AGENT_GAP  = 0.5

# Per-metric y-axis limits: display_name -> (y_min, y_max)
# Leave a metric out (or set to None) to use automatic limits.
YLIM_MAP: dict[str, tuple[float, float] | None] = {
    "CC":                  (-2,2),
    "WMC":                 (-20,40),
    "MaxNest":             (-5,10),
    "NPATH":               (-3,5),
    "CBO":                 (-5,5),
    "RFC":                 (-10,10),
    "NOC":                 (-2.5,2.5),
    "IFANIN":              (-1.5,5.5),
    "Declarative Methods": (-2,5),
    "NL":                  (-100,150),
    "LOC":                 (-75,125),
    "BLOC":                (-20,30),
    "CLOC":                (-30,60),
    "STMTC":               (-30,70),
    "Classes":             (-1.5,5),
    "Functions":           (-3,10),
    "NIM":                 (-2,3),
    "NIV":                 (-2,3),
}

for attr, metrics in METRICS_MAP.items():
    for display_name, level, raw_metric in metrics:
        model_data = []

        for model_name, df in data_by_model.items():
            h_col = f"diff_{raw_metric}_{level}_human"
            l_col = f"diff_{raw_metric}_{level}_llm"
            if h_col not in df.columns or l_col not in df.columns:
                continue
            human_vals = remove_outliers(pd.to_numeric(df[h_col], errors="coerce")).dropna().values
            llm_vals   = remove_outliers(pd.to_numeric(df[l_col], errors="coerce")).dropna().values
            x_label    = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            model_data.append((x_label, human_vals, llm_vals, model_name))

        if not model_data:
            continue

        n           = len(model_data)
        box_w       = 0.35   # width of each box
        group_gap   = 0.9    # x-distance between group centres

        fig, ax = plt.subplots(figsize=(n * group_gap + 0.6, 2.2))

        gold_patch  = mpatches.Patch(facecolor=GOLD_COLOR, alpha=0.9, label="Gold")
        model_patch = mpatches.Patch(facecolor=LLM_COLOR,  alpha=0.9, label="LLM")

        all_whisker_vals = []

        for i, (x_label, human_vals, llm_vals, model_name) in enumerate(model_data):
            cx       = i * group_gap
            pos_h    = cx - box_w / 2
            pos_l    = cx + box_w / 2

            bp_h = ax.boxplot(human_vals, positions=[pos_h], widths=box_w,
                              patch_artist=True, notch=False, showfliers=True,
                              medianprops=dict(color="black", linewidth=1.5),
                              whiskerprops=dict(linewidth=1.0),
                              capprops=dict(linewidth=1.0),
                              boxprops=dict(linewidth=1.0),
                              flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                                              markerfacecolor=GOLD_COLOR, alpha=0.5))
            bp_h["boxes"][0].set_facecolor(GOLD_COLOR)
            bp_h["boxes"][0].set_alpha(0.9)

            bp_l = ax.boxplot(llm_vals, positions=[pos_l], widths=box_w,
                              patch_artist=True, notch=False, showfliers=True,
                              medianprops=dict(color="black", linewidth=1.5),
                              whiskerprops=dict(linewidth=1.0),
                              capprops=dict(linewidth=1.0),
                              boxprops=dict(linewidth=1.0),
                              flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                                              markerfacecolor=LLM_COLOR, alpha=0.5))
            bp_l["boxes"][0].set_facecolor(LLM_COLOR)
            bp_l["boxes"][0].set_alpha(0.9)

            for w in bp_h["whiskers"] + bp_l["whiskers"]:
                all_whisker_vals.extend(w.get_ydata())

            # Dashed separator between model groups
            if i < n - 1:
                ax.axvline(cx + group_gap / 2, color="grey",
                           linestyle="--", linewidth=0.8, alpha=0.7)

        # Y-axis limits
        ylim = YLIM_MAP.get(display_name)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        elif all_whisker_vals:
            w_min, w_max = min(all_whisker_vals), max(all_whisker_vals)
            if w_min == w_max:
                combined = np.concatenate([v for _, hv, lv, _ in model_data
                                           for v in [hv, lv]])
                combined = combined[np.isfinite(combined)]
                if combined.size:
                    w_min = np.percentile(combined, 5)
                    w_max = np.percentile(combined, 95)
            pad = (w_max - w_min) * 0.20 or 0.5
            ax.set_ylim(w_min - pad, w_max + pad)

        # X-axis: model name labels centred on each group
        ax.set_xticks([i * group_gap for i in range(n)])
        ax.set_xticklabels([x_label for x_label, _, _, _ in model_data], fontsize=8)
        ax.set_xlim(-group_gap * 0.5, (n - 1) * group_gap + group_gap * 0.5)
        #ax.set_ylabel(display_name, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        # fig.legend(handles=[gold_patch, model_patch],
        #            loc="upper center", bbox_to_anchor=(0.5, 1.06),
        #            ncol=2, frameon=False, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        safe = f"{attr}_{display_name}_{level}".replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(str(BOX_DIR / f"{safe}.pdf"), bbox_inches="tight")
        plt.close(fig)

print("  Done.")


# ── Combined LaTeX table ────────────────────────────────────────────────────────

print("\nGenerating combined LaTeX table ...")

central_label = "median" if USE_MEDIAN else "mean"

all_metrics_ordered = [
    (attr, display_name, level)
    for attr, metrics in METRICS_MAP.items()
    for display_name, level, _ in metrics
]

n_models    = len(stats_by_model)
model_names = list(stats_by_model.keys())

model_header_parts = " & ".join(
    rf"\multicolumn{{5}}{{c}}{{\textbf{{{MODEL_DISPLAY_NAMES.get(m, m)}}}}}" for m in model_names
)
sub_parts = " & ".join(
    r"\textbf{Gold} & \textbf{LLM} & \textbf{$p$} & \textbf{$\delta$} & \textbf{Mag.}"
    for _ in model_names
)
col_spec = "ll " + " ".join(["r r r r c"] * n_models)

lines = []
lines.append("% Add to LaTeX preamble:")
lines.append(r"% \usepackage[table]{xcolor}")
lines.append(r"% \definecolor{magnitudeS}{HTML}{FFF176}  % small  – light yellow")
lines.append(r"% \definecolor{magnitudeM}{HTML}{FFB74D}  % medium – orange")
lines.append(r"% \definecolor{magnitudeL}{HTML}{E57373}  % large  – red")
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\setlength{\tabcolsep}{3pt}")
lines.append(r"\renewcommand{\arraystretch}{1.1}")
lines.append(
    rf"\caption{{All models: LLM vs Gold patch metric deltas — "
    rf"{central_label}, Wilcoxon $p$-value, and Cliff's $\delta$ effect size.}}"
)
lines.append(r"\label{tab:all_models_llm_vs_gold}")
lines.append(r"\resizebox{0.9\textwidth}{!}{")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
lines.append(
    rf"\textbf{{Quality Attr.}} & \textbf{{Metric}} & {model_header_parts} \\"
)
cmidrules = " ".join(
    rf"\cmidrule(lr){{{3 + i*5}-{7 + i*5}}}" for i in range(n_models)
)
lines.append(cmidrules)
lines.append(rf"\textbf{{}} & \textbf{{}} & {sub_parts} \\")
lines.append(r"\midrule")

prev_attr = None
for attr, display_name, level in all_metrics_ordered:
    attr_metrics    = [dn for a, dn, _ in all_metrics_ordered if a == attr]
    attr_total_rows = len(attr_metrics)

    if attr != prev_attr:
        lines.append(r"\midrule")
        prev_attr = attr
        attr_first_in_block = True
    else:
        attr_first_in_block = False

    metric_idx = attr_metrics.index(display_name)

    attr_cell   = rf"\multirow{{{attr_total_rows}}}{{*}}{{{attr}}}" if attr_first_in_block else ""
    metric_cell = display_name

    model_cols = []
    for model_name in model_names:
        df_res = stats_by_model.get(model_name)
        if df_res is None:
            model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
            continue
        row = df_res[df_res["metric"] == display_name]
        if row.empty:
            model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--} & \text{--}")
        else:
            r = row.iloc[0]
            model_cols.append(
                f"{fmt_val(r['gold_central'])} & {fmt_val(r['model_central'])} & "
                f"{fmt_p(r['p'])} & {r['cliffs_delta']:.3f} & {fmt_mag(r['delta_mag'])}"
            )

    lines.append(
        f"{attr_cell} & {metric_cell} & "
        + " & ".join(model_cols) + r" \\"
    )

    if metric_idx < len(attr_metrics) - 1:
        end_col = 2 + n_models * 5
        lines.append(rf"\cmidrule(lr){{2-{end_col}}}")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")
lines.append(r"\end{table*}")

tex_path = OUT_DIR / "comparison_table_all_models.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  LaTeX table → {tex_path.name}")

print("\nDone.")