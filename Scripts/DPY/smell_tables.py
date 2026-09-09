"""
smell_tables.py — Generate one LaTeX table per smell category
(arch_smells, design_smells, implementation_smells).

Structure per table:
  Rows:    smell type x change direction (I/R)  [all instances, no size band]
  Columns: model groups, each with (H, A, p, Mag)
             H   = Gold mean
             A   = LLM mean
             p   = Wilcoxon p-value (bold + * if significant)
             Mag = Cliff's delta magnitude

Outputs (dpy_results/results/):
  smell_table_arch.tex
  smell_table_design.tex
  smell_table_impl.tex

Usage:
    python smell_tables.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).parent
AGGREGATED_CSV = _HERE / "dpy_results/aggregated.csv"
OUT_DIR        = _HERE / "dpy_results/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model display names (in desired column order) ──────────────────────────────
MODELS = [
    ("claude-45haiku-10222025",  "Haiku"),
    ("claude-45sonnet-10132025", "Sonnet"),
    ("codex",                    "Codex"),
    ("gemini-2-5-pro-nov17",     "Gemini"),
    ("glm-4p5-10222025",         "GLM"),
    ("gpt-5-codex-debug-oct22",  "GPT-5D"),
    ("gpt-5-high-paper",         "GPT-5H"),
    ("gptoss-paper",             "GPT-OSS"),
]

MAG_SHORT = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}

# ── Smell definitions ──────────────────────────────────────────────────────────
# (table_label, caption, list of (display_name, col_stem))
#   col_stem -> aggregated columns are gold_{prefix}_{col_stem}_added / _removed
SMELL_GROUPS = [
    (
        "arch",
        "Arch Smells Analysis by Patch Size Band",
        "tab:arch-smells",
        "arch",
        [
            ("God Component",       "god_component"),
            ("Unstable Dependency", "unstable_dependency"),
        ],
    ),
    (
        "design",
        "Design Smells Analysis by Patch Size Band",
        "tab:design-smells",
        "design",
        [
            ("Broken Hierarchy",             "broken_hierarchy"),
            ("Broken Modularization",        "broken_modularization"),
            ("Deficient Encapsulation",      "deficient_encapsulation"),
            ("Feature Envy",                 "feature_envy"),
            ("Insufficient Modularization",  "insufficient_modularization"),
            ("Multifaceted Abstraction",     "multifaceted_abstraction"),
            ("Rebellious Hierarchy",         "rebellious_hierarchy"),
            ("Wide Hierarchy",               "wide_hierarchy"),
        ],
    ),
    (
        "impl",
        "Implementation Smells Analysis by Patch Size Band",
        "tab:impl-smells",
        "impl",
        [
            ("Complex Conditional",  "complex_conditional"),
            ("Complex Method",       "complex_method"),
            ("Empty Catch Block",    "empty_catch_block"),
            ("Long Identifier",      "long_identifier"),
            ("Long Lambda Function", "long_lambda_function"),
            ("Long Method",          "long_method"),
            ("Long Parameter List",  "long_parameter_list"),
            ("Long Statement",       "long_statement"),
            ("Magic Number",         "magic_number"),
            ("Missing Default",      "missing_default"),
        ],
    ),
]


# ── Statistical helpers ────────────────────────────────────────────────────────

def run_test(gold: pd.Series, llm: pd.Series):
    """
    Wilcoxon signed-rank test on paired (gold, llm) counts.
    Returns (gold_mean, llm_mean, p, mag_short).
    """
    pairs = pd.DataFrame({"g": gold, "l": llm}).dropna()
    h_mean = float(pairs["g"].mean()) if not pairs.empty else np.nan
    l_mean = float(pairs["l"].mean()) if not pairs.empty else np.nan

    if len(pairs) < 5 or (pairs["g"] == pairs["l"]).all():
        return h_mean, l_mean, np.nan, "N"

    try:
        _, p = wilcoxon(pairs["l"], pairs["g"], zero_method="pratt", alternative="two-sided")
    except Exception:
        return h_mean, l_mean, np.nan, "N"

    cd, mag = cliffs_delta_lib(pairs["l"].values.tolist(), pairs["g"].values.tolist())
    return h_mean, l_mean, p, MAG_SHORT.get(mag, mag[0].upper())


def fmt_mean(v):
    if np.isnan(v):
        return r"\text{--}"
    return f"{v:.2f}"


def fmt_p(p):
    if np.isnan(p):
        return r"\text{--}"
    stars = ""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    txt = f"{p:.3f}{stars}"
    return rf"\textbf{{{txt}}}" if stars else txt


def fmt_mag(mag):
    return f"({mag})"


# ── Load data ──────────────────────────────────────────────────────────────────

print(f"Loading {AGGREGATED_CSV} ...")
df_all = pd.read_csv(AGGREGATED_CSV)
print(f"  {len(df_all)} rows loaded")

# Split by model for fast lookup
model_dfs = {m: df_all[df_all["model"] == m].copy() for m, _ in MODELS}


# ── Table generator ────────────────────────────────────────────────────────────

def build_table(file_key, caption, label, prefix, smell_list):
    n_models   = len(MODELS)
    col_spec   = "l c " + " ".join(["rrrr"] * n_models)
    model_span = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{\textbf{{{disp}}}}}" for _, disp in MODELS
    )
    cmidrules = "".join(
        rf"\cmidrule(lr){{{3 + i*4}-{6 + i*4}}}" for i in range(n_models)
    )
    sub_header = " & ".join(r"H & A & $p$ & Mag" for _ in MODELS)

    lines = []
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(
        rf"\multicolumn{{2}}{{c}}{{}} & {model_span} \\"
    )
    lines.append(cmidrules)
    lines.append(
        rf"Metric & Change & {sub_header} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\midrule")

    for smell_display, col_stem in smell_list:
        first_smell = True

        for change, direction in [("added", "I"), ("removed", "R")]:
            gold_col = f"gold_{prefix}_{col_stem}_{change}"
            llm_col  = f"llm_{prefix}_{col_stem}_{change}"

            smell_cell = (
                rf"\multirow{{2}}{{*}}{{{smell_display}}}"
                if first_smell
                else ""
            )

            model_cols = []
            for model_key, _ in MODELS:
                df = model_dfs.get(model_key)
                if df is None or gold_col not in df.columns:
                    model_cols.append(r"\text{--} & \text{--} & \text{--} & \text{--}")
                    continue
                h, a, p, mag = run_test(
                    pd.to_numeric(df[gold_col], errors="coerce"),
                    pd.to_numeric(df[llm_col],  errors="coerce"),
                )
                model_cols.append(
                    f"{fmt_mean(h)} & {fmt_mean(a)} & {fmt_p(p)} & {fmt_mag(mag)}"
                )

            lines.append(
                f"{smell_cell} & {direction} & "
                + " & ".join(model_cols) + r" \\"
            )
            first_smell = False

        lines.append(r"\midrule")

    # Remove last \midrule, replace with \bottomrule
    lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item \textbf{Notes:} H = Human (Gold) mean, A = Agent (LLM) mean, $p$ = Wilcoxon p-value, Mag = Cliff's $\delta$ magnitude")
    lines.append(r"\item Significance: * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$")
    lines.append(r"\item Magnitude: N = Negligible, S = Small, M = Medium, L = Large")
    lines.append(r"\item Change: I = Introduced (added), R = Removed")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Generate all tables ────────────────────────────────────────────────────────

for file_key, caption, label, prefix, smell_list in SMELL_GROUPS:
    tex = build_table(file_key, caption, label, prefix, smell_list)
    out_path = OUT_DIR / f"smell_table_{file_key}.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"  Written -> {out_path.name}")

print("\nDone.")
