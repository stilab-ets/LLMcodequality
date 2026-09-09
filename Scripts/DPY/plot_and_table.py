"""
plot_and_table.py

Compare each model's LLM patch vs its own Gold (human) patch DPy metric diffs per instance.
Data comes from dpy_results/aggregated.csv (one row per instance, gold_* and llm_* columns).

All models are shown together in combined plots and a single combined LaTeX table.

Flags:
  --central mean|median      Central tendency for the table (default: median)
  --remove-outliers          Clip values outside [0.5%, 99.5%] before all analysis
  --outlier-threshold FLOAT  Percentile threshold for clipping (default: 0.5)
  --show-fliers              Show outlier points in box plots (default: hidden)

Outputs (written to dpy_results/results/):
  - One PDF box plot per metric (all models, gold vs llm side-by-side) -> results/boxplots/
  - Combined LaTeX table (all models, each llm vs its own gold)        -> results/comparison_table_all_models.tex
  - Per-model stats CSV                                                -> results/comparison_stats_<model>.csv
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
parser.add_argument("--central", choices=["mean", "median"], default="median")
parser.add_argument("--remove-outliers", action="store_true", default=False)
parser.add_argument("--outlier-threshold", type=float, default=0.5)
parser.add_argument("--show-fliers", action="store_true", default=False)
args = parser.parse_args()
USE_MEDIAN      = args.central == "median"
REMOVE_OUTLIERS = args.remove_outliers
OUTLIER_PCT     = args.outlier_threshold / 100.0
SHOW_FLIERS     = args.show_fliers

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).parent
AGGREGATED_CSV = _HERE / "dpy_results/aggregated.csv"
PATCH_SIZE_CSV = Path(r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\understand_results\patch_size_lookup.csv")
OUT_DIR        = _HERE / "dpy_results/results"
BOX_DIR        = OUT_DIR / "boxplots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOX_DIR.mkdir(parents=True, exist_ok=True)

# ── Model display names & colours ─────────────────────────────────────────────
GOLD_COLOR = "#f4a261"
LLM_COLOR  = "#4c72b0"

MODEL_COLORS = {
    "claude-45haiku-10222025":  "#1f77b4",
    "claude-45sonnet-10132025": "#ff7f0e",
    "codex":                    "#2ca02c",
    "gemini-2-5-pro-nov17":     "#d62728",
    "glm-4p5-10222025":         "#9467bd",
    #"gpt-5-codex-debug-oct22":  "#8c564b",
    "gpt-5-high-paper":         "#e377c2",
    "gptoss-paper":             "#7f7f7f",
}

MODEL_DISPLAY_NAMES = {
    "claude-45haiku-10222025":  "Haiku",
    "claude-45sonnet-10132025": "Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    #"gpt-5-codex-debug-oct22":  "GPT-5D",
    "gpt-5-high-paper":         "GPT-5H",
    "gptoss-paper":             "GPT-OSS",
}


# ── Metrics: (display_name, gold_col, llm_col) ────────────────────────────────
METRICS_MAP = {
    "Function Metrics": [
        ("LOC",  "gold_fn_LOC",  "llm_fn_LOC"),
        ("CC",   "gold_fn_CC",   "llm_fn_CC"),
        ("PC",   "gold_fn_PC",   "llm_fn_PC"),
    ],
    "Class / Module Metrics": [
        ("LOC",     "gold_cls_LOC",     "llm_cls_LOC"),
        ("WMC",     "gold_cls_WMC",     "llm_cls_WMC"),
        ("NOM",     "gold_cls_NOM",     "llm_cls_NOM"),
        ("NOPM",    "gold_cls_NOPM",    "llm_cls_NOPM"),
        ("NOF",     "gold_cls_NOF",     "llm_cls_NOF"),
        ("NOPF",    "gold_cls_NOPF",    "llm_cls_NOPF"),
        ("LCOM",    "gold_cls_LCOM",    "llm_cls_LCOM"),
        ("Fan-In",  "gold_cls_Fan-In",  "llm_cls_Fan-In"),
        ("Fan-Out", "gold_cls_Fan-Out", "llm_cls_Fan-Out"),
        ("DIT",     "gold_cls_DIT",     "llm_cls_DIT"),
    ],
    # "Arch Smells": [
    #     ("God component",       "gold_arch_god_component_added",         "llm_arch_god_component_added"),
    #     ("God component (-)",   "gold_arch_god_component_removed",       "llm_arch_god_component_removed"),
    #     ("Unstable dep.",       "gold_arch_unstable_dependency_added",   "llm_arch_unstable_dependency_added"),
    #     ("Unstable dep. (-)",   "gold_arch_unstable_dependency_removed", "llm_arch_unstable_dependency_removed"),
    # ],
    # "Design Smells": [
    #     ("Broken hier.",        "gold_design_broken_hierarchy_added",              "llm_design_broken_hierarchy_added"),
    #     ("Broken hier. (-)",    "gold_design_broken_hierarchy_removed",            "llm_design_broken_hierarchy_removed"),
    #     ("Broken mod.",         "gold_design_broken_modularization_added",         "llm_design_broken_modularization_added"),
    #     ("Broken mod. (-)",     "gold_design_broken_modularization_removed",       "llm_design_broken_modularization_removed"),
    #     ("Deficient enc.",      "gold_design_deficient_encapsulation_added",       "llm_design_deficient_encapsulation_added"),
    #     ("Deficient enc. (-)",  "gold_design_deficient_encapsulation_removed",     "llm_design_deficient_encapsulation_removed"),
    #     ("Feature envy",        "gold_design_feature_envy_added",                 "llm_design_feature_envy_added"),
    #     ("Feature envy (-)",    "gold_design_feature_envy_removed",               "llm_design_feature_envy_removed"),
    #     ("Insuf. mod.",         "gold_design_insufficient_modularization_added",   "llm_design_insufficient_modularization_added"),
    #     ("Insuf. mod. (-)",     "gold_design_insufficient_modularization_removed", "llm_design_insufficient_modularization_removed"),
    #     ("Multifaceted abs.",   "gold_design_multifaceted_abstraction_added",      "llm_design_multifaceted_abstraction_added"),
    #     ("Multifaceted abs.(-)", "gold_design_multifaceted_abstraction_removed",   "llm_design_multifaceted_abstraction_removed"),
    #     ("Rebellious hier.",    "gold_design_rebellious_hierarchy_added",          "llm_design_rebellious_hierarchy_added"),
    #     ("Rebellious hier.(-)", "gold_design_rebellious_hierarchy_removed",        "llm_design_rebellious_hierarchy_removed"),
    #     ("Wide hier.",          "gold_design_wide_hierarchy_added",                "llm_design_wide_hierarchy_added"),
    #     ("Wide hier. (-)",      "gold_design_wide_hierarchy_removed",              "llm_design_wide_hierarchy_removed"),
    # ],
    # "Impl Smells": [
    #     ("Complex cond.",       "gold_impl_complex_conditional_added",      "llm_impl_complex_conditional_added"),
    #     ("Complex cond. (-)",   "gold_impl_complex_conditional_removed",    "llm_impl_complex_conditional_removed"),
    #     ("Complex method",      "gold_impl_complex_method_added",           "llm_impl_complex_method_added"),
    #     ("Complex method (-)",  "gold_impl_complex_method_removed",         "llm_impl_complex_method_removed"),
    #     ("Empty catch",         "gold_impl_empty_catch_block_added",        "llm_impl_empty_catch_block_added"),
    #     ("Empty catch (-)",     "gold_impl_empty_catch_block_removed",      "llm_impl_empty_catch_block_removed"),
    #     ("Long identifier",     "gold_impl_long_identifier_added",          "llm_impl_long_identifier_added"),
    #     ("Long identifier (-)", "gold_impl_long_identifier_removed",        "llm_impl_long_identifier_removed"),
    #     ("Long lambda",         "gold_impl_long_lambda_function_added",     "llm_impl_long_lambda_function_added"),
    #     ("Long lambda (-)",     "gold_impl_long_lambda_function_removed",   "llm_impl_long_lambda_function_removed"),
    #     ("Long method",         "gold_impl_long_method_added",              "llm_impl_long_method_added"),
    #     ("Long method (-)",     "gold_impl_long_method_removed",            "llm_impl_long_method_removed"),
    #     ("Long param list",     "gold_impl_long_parameter_list_added",      "llm_impl_long_parameter_list_added"),
    #     ("Long param list (-)", "gold_impl_long_parameter_list_removed",    "llm_impl_long_parameter_list_removed"),
    #     ("Long statement",      "gold_impl_long_statement_added",           "llm_impl_long_statement_added"),
    #     ("Long statement (-)",  "gold_impl_long_statement_removed",         "llm_impl_long_statement_removed"),
    #     ("Magic number",        "gold_impl_magic_number_added",             "llm_impl_magic_number_added"),
    #     ("Magic number (-)",    "gold_impl_magic_number_removed",           "llm_impl_magic_number_removed"),
    #     ("Missing default",     "gold_impl_missing_default_added",          "llm_impl_missing_default_added"),
    #     ("Missing default (-)", "gold_impl_missing_default_removed",        "llm_impl_missing_default_removed"),
    # ],
}

# Per-metric y-axis limits: display_name -> (y_min, y_max) or None for auto.
YLIM_MAP: dict[str, tuple[float, float] | None] = {
    # Function Metrics
    "LOC":                  None,
    "CC":                   None,
    "PC":                   None,
    # Class / Module Metrics
    "WMC":                  (-40,70),
    "NOM":                  (-5,10),
    "NOPM":                 (-5,5),
    "NOF":                  (-5,6),
    "NOPF":                 (-4,5),
    "LCOM":                 (-1,1),
    "Fan-In":               (-2,5),
    "Fan-Out":              (-5,5),
    "DIT":                  (-1.5,5),
}

# ── Statistical helpers ────────────────────────────────────────────────────────
MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}


def remove_outliers_fn(series: pd.Series) -> pd.Series:
    if not REMOVE_OUTLIERS:
        return series
    lo = series.quantile(OUTLIER_PCT)
    hi = series.quantile(1.0 - OUTLIER_PCT)
    return series.where((series >= lo) & (series <= hi))


def run_test(human_vals, llm_vals):
    """Wilcoxon signed-rank on paired (gold, llm) diffs after dropping NaNs."""
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


# ── Load & merge data ──────────────────────────────────────────────────────────
print(f"Loading {AGGREGATED_CSV} ...")
df_all = pd.read_csv(AGGREGATED_CSV)
print(f"  {len(df_all)} rows, {len(df_all.columns)} columns")


# ── Per-model stats ────────────────────────────────────────────────────────────
data_by_model:  dict[str, pd.DataFrame] = {}
stats_by_model: dict[str, pd.DataFrame] = {}

for model_name in sorted(df_all["model"].unique()):
    df = df_all[df_all["model"] == model_name].copy()
    data_by_model[model_name] = df
    print(f"  {model_name}: {len(df)} instances")

    rows = []
    for attr, metrics in METRICS_MAP.items():
        for display_name, gold_col, llm_col in metrics:
            if gold_col not in df.columns or llm_col not in df.columns:
                continue
            p, h_c, l_c, cd, dmag = run_test(
                remove_outliers_fn(pd.to_numeric(df[gold_col], errors="coerce")),
                remove_outliers_fn(pd.to_numeric(df[llm_col],  errors="coerce")),
            )
            rows.append({
                "attr": attr, "metric": display_name,
                "gold_central": h_c, "model_central": l_c,
                "p": p, "cliffs_delta": cd, "delta_mag": dmag,
            })

    df_res = pd.DataFrame(rows)
    stats_by_model[model_name] = df_res
    df_res.to_csv(OUT_DIR / f"comparison_stats_{model_name}.csv", index=False)
    print(f"    stats saved -> comparison_stats_{model_name}.csv")

# ── Box plots ──────────────────────────────────────────────────────────────────
print(f"\nGenerating box plots -> {BOX_DIR}/")

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 14,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

for attr, metrics in METRICS_MAP.items():
    for display_name, gold_col, llm_col in metrics:
        model_data = []
        for model_name, df in data_by_model.items():
            if model_name not in MODEL_DISPLAY_NAMES:
                continue
            if gold_col not in df.columns or llm_col not in df.columns:
                continue
            human_vals = remove_outliers_fn(pd.to_numeric(df[gold_col], errors="coerce")).dropna().values
            llm_vals   = remove_outliers_fn(pd.to_numeric(df[llm_col],  errors="coerce")).dropna().values
            model_data.append((MODEL_DISPLAY_NAMES.get(model_name, model_name), human_vals, llm_vals, model_name))

        if not model_data:
            continue

        n           = len(model_data)
        box_w       = 0.35
        group_gap   = 0.9

        fig, ax = plt.subplots(figsize=(n * group_gap + 0.6, 2.2))

        gold_patch  = mpatches.Patch(facecolor=GOLD_COLOR, alpha=0.9, label="Gold")
        model_patch = mpatches.Patch(facecolor=LLM_COLOR,  alpha=0.9, label="LLM")

        all_whisker_vals = []

        for i, (x_label, human_vals, llm_vals, model_name) in enumerate(model_data):
            cx    = i * group_gap
            pos_h = cx - box_w / 2
            pos_l = cx + box_w / 2

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

        ax.set_xticks([i * group_gap for i in range(n)])
        ax.set_xticklabels([x_label for x_label, _, _, _ in model_data], fontsize=8)
        ax.set_xlim(-group_gap * 0.5, (n - 1) * group_gap + group_gap * 0.5)
        # ax.set_ylabel(display_name, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        # fig.legend(handles=[gold_patch, model_patch],
        #            loc="upper center", bbox_to_anchor=(0.5, 1.06),
        #            ncol=2, frameon=False, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        safe = f"{attr}_{display_name}".replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace("-", "")
        fig.savefig(str(BOX_DIR / f"{safe}.pdf"), bbox_inches="tight")
        plt.close(fig)

print("  Done.")

# ── Combined LaTeX table ───────────────────────────────────────────────────────
print("\nGenerating combined LaTeX table ...")

central_label = "median" if USE_MEDIAN else "mean"
model_names   = list(stats_by_model.keys())
n_models      = len(model_names)

all_metrics_ordered = [
    (attr, display_name, gold_col, llm_col)
    for attr, metrics in METRICS_MAP.items()
    for display_name, gold_col, llm_col in metrics
]

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
    rf"\caption{{All models: LLM vs Gold patch DPy metric deltas (all instances) --- "
    rf"{central_label}, Wilcoxon $p$-value, and Cliff's $\delta$ effect size.}}"
)
lines.append(r"\label{tab:all_models_dpy_llm_vs_gold}")
lines.append(r"\resizebox{0.9\textwidth}{!}{")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
lines.append(
    rf"\textbf{{Quality Attr.}} & \textbf{{Metric}} & {model_header_parts} \\"
)
cmidrules = " ".join(
    rf"\cmidrule(lr){{{3 + i * 5}-{7 + i * 5}}}" for i in range(n_models)
)
lines.append(cmidrules)
lines.append(rf"\textbf{{}} & \textbf{{}} & {sub_parts} \\")
lines.append(r"\midrule")

prev_attr = None
for attr, display_name, gold_col, llm_col in all_metrics_ordered:
    attr_metrics    = [dn for a, dn, _, __ in all_metrics_ordered if a == attr]
    attr_total_rows = len(attr_metrics)

    if attr != prev_attr:
        lines.append(r"\midrule")
        prev_attr  = attr
        attr_first = True
    else:
        attr_first = False

    metric_idx = attr_metrics.index(display_name)

    attr_cell   = (rf"\multirow{{{attr_total_rows}}}{{*}}{{{attr}}}" if attr_first else "")

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
        f"{attr_cell} & {display_name} & "
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
print(f"  LaTeX table -> {tex_path}")

print("\nDone.")
