#!/usr/bin/env python3
"""
rq_patch_structure_table.py

Generates a LaTeX table comparing patch-level structural metrics between
agent (LLM) patches and human (gold) patches, using data from python_data/succeeded/.

Metrics computed from each diff file:
  Lines Added, Lines Deleted, Net Change, Change Ratio (add/total),
  Files Changed, Entropy, Unique Directories, Top-Level Directories, Max Depth

Output → Scripts/results/rq/rq_table_patch_structure.tex
"""

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta as cliffs_delta_lib
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
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

# ── Diff parser ────────────────────────────────────────────────────────────────

def parse_diff(path: Path) -> dict:
    """Extract patch-level metrics from a unified diff file."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None

    files = re.findall(r"^diff --git a/(.+?) b/", text, re.MULTILINE)
    if not files:
        return None

    added = deleted = 0
    file_changes: dict[str, int] = {}

    current_file = None
    for line in text.splitlines():
        if line.startswith("diff --git a/"):
            m = re.match(r"diff --git a/(.+?) b/", line)
            if m:
                current_file = m.group(1)
                file_changes.setdefault(current_file, 0)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
            if current_file:
                file_changes[current_file] = file_changes.get(current_file, 0) + 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            if current_file:
                file_changes[current_file] = file_changes.get(current_file, 0) + 1

    total_changes = sum(file_changes.values())
    if total_changes > 0:
        probs = [c / total_changes for c in file_changes.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0

    dirs = {str(Path(f).parent) for f in file_changes}
    top_dirs = {Path(f).parts[0] if len(Path(f).parts) > 1 else "." for f in file_changes}
    depths = [len(Path(f).parts) for f in file_changes]

    return {
        "lines_added":    added,
        "lines_deleted":  deleted,
        "net_change":     added - deleted,
        "change_ratio":   added / (added + deleted) if (added + deleted) > 0 else 0.0,
        "files_changed":  len(file_changes),
        "entropy":        entropy,
        "unique_dirs":    len(dirs),
        "top_dirs":       len(top_dirs),
        "max_depth":      max(depths) if depths else 0,
    }

# ── Collect per-instance metrics ───────────────────────────────────────────────

METRIC_KEYS = [
    ("Lines Added",              "lines_added"),
    ("Lines Deleted",            "lines_deleted"),
    ("Net Change",               "net_change"),
    ("Change Ratio (add/total)", "change_ratio"),
    ("Files Changed",            "files_changed"),
    ("Entropy",                  "entropy"),
    ("Unique Directories",       "unique_dirs"),
    ("Top-Level Directories",    "top_dirs"),
    ("Max Depth",                "max_depth"),
]

records = []
for model in MODELS:
    model_dir = SUCCEEDED / model
    if not model_dir.exists():
        print(f"  SKIP {model} — directory not found")
        continue
    for instance_dir in sorted(model_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        llm_diff  = instance_dir / "filtered_llm.diff"
        gold_diff = instance_dir / "filtered_gold.diff"
        if not llm_diff.exists():
            llm_diff = instance_dir / "_patch.diff"
        llm_stats  = parse_diff(llm_diff)
        gold_stats = parse_diff(gold_diff)
        if llm_stats is None or gold_stats is None:
            continue
        records.append({
            "model":    model,
            "instance": instance_dir.name,
            **{f"llm_{k}":  v for k, v in llm_stats.items()},
            **{f"gold_{k}": v for k, v in gold_stats.items()},
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} instances across {df['model'].nunique()} models")
df.to_csv(OUT_DIR / "rq_patch_structure_data.csv", index=False)

# ── Statistical helpers ────────────────────────────────────────────────────────
MAG_FULL = {"negligible": "N", "small": "S", "medium": "M", "large": "L"}


def stars(p):
    if p is None or np.isnan(p): return ""
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def run_test(gold: pd.Series, llm: pd.Series):
    pairs = pd.DataFrame({"g": gold, "l": llm}).dropna()
    n = len(pairs)
    if n < 10:
        return n, np.nan, np.nan, np.nan, "N"
    d = pairs["l"] - pairs["g"]
    g_mean = pairs["g"].median()
    l_mean = pairs["l"].median()
    if (d == 0).all():
        return n, g_mean, l_mean, 1.0, "N"
    try:
        _, p = wilcoxon(pairs["l"], pairs["g"], zero_method="pratt",
                        alternative="two-sided")
    except Exception:
        p = np.nan
    mag_label, mag_str = cliffs_delta_lib(
        pairs["l"].values.tolist(), pairs["g"].values.tolist()
    )
    return n, g_mean, l_mean, p, MAG_FULL.get(mag_str, mag_str[0].upper())


def bh_correct(pvals: list, alpha: float = 0.05) -> list:
    """Benjamini-Hochberg correction. Returns adjusted p-values (same length)."""
    n = len(pvals)
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    if not valid:
        return pvals[:]
    sorted_valid = sorted(valid, key=lambda x: x[1])
    m = len(sorted_valid)
    p_adj = list(pvals)
    # compute BH-adjusted p-value: p_adj_(i) = p_(i) * m / rank, made monotone
    adj = [np.nan] * m
    for rank, (orig_i, p) in enumerate(sorted_valid):
        adj[rank] = min(p * m / (rank + 1), 1.0)
    # enforce monotonicity (non-decreasing in rank order)
    for rank in range(m - 2, -1, -1):
        adj[rank] = min(adj[rank], adj[rank + 1])
    for rank, (orig_i, _) in enumerate(sorted_valid):
        p_adj[orig_i] = adj[rank]
    return p_adj


def fmt_val(v):
    if v is None or np.isnan(v): return r"\text{--}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 10:   return f"{v:.2f}"
    return f"{v:.3f}"


MAG_COLORS = {"S": "magnitudeS", "M": "magnitudeM", "L": "magnitudeL"}

def fmt_sig(mag, p):
    s = stars(p)
    cell = rf"$\text{{{mag}{s}}}$"
    if mag in MAG_COLORS:
        return rf"\cellcolor{{{MAG_COLORS[mag]}}}{cell}"
    return cell

# ── First pass: collect all raw p-values for BH correction ────────────────────

active_models = [m for m in MODELS if m in df["model"].values]
n_models = len(active_models)

# stat_grid[(metric_key, model)] = (n, g_med, l_med, raw_p, mag)
stat_grid: dict = {}
for display, key in METRIC_KEYS:
    for m in active_models:
        mdf = df[df["model"] == m]
        if mdf.empty:
            stat_grid[(key, m)] = (0, np.nan, np.nan, np.nan, "N")
        else:
            stat_grid[(key, m)] = run_test(mdf[f"gold_{key}"], mdf[f"llm_{key}"])

# Collect p-values in a fixed order and apply BH across all 72 tests
cell_keys = [(key, m) for display, key in METRIC_KEYS for m in active_models]
raw_pvals = [stat_grid[k][3] for k in cell_keys]
adj_pvals = bh_correct(raw_pvals)
adj_p_map = {k: adj_pvals[i] for i, k in enumerate(cell_keys)}

n_raw_sig = sum(1 for p in raw_pvals if p is not None and not np.isnan(p) and p < 0.05)
n_adj_sig = sum(1 for p in adj_pvals if p is not None and not np.isnan(p) and p < 0.05)
print(f"BH correction: {n_raw_sig} significant before → {n_adj_sig} after (α=0.05, {len(raw_pvals)} tests)")

# ── Build LaTeX table ──────────────────────────────────────────────────────────

col_spec      = "l" + "rrc" * n_models
model_headers = " & ".join(
    rf"\multicolumn{{3}}{{c}}{{\textbf{{{MODEL_LABELS[m]}}}}}"
    for m in active_models
)
cmidrule = " ".join(
    rf"\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}"
    for i in range(n_models)
)
sub_header = " & ".join(
    r"\textbf{Agent} & \textbf{Human} & \textbf{Sig.}"
    for _ in active_models
)

# N row
n_cells = []
for m in active_models:
    n_cells.append(rf"\multicolumn{{3}}{{c}}{{{len(df[df['model'] == m])}}}")

sig_note = (
    r"Patch-level structural metrics comparison (median values). "
    r"Agent values annotated with Cliff's $\delta$ magnitude "
    r"(\textbf{N}=negligible, \textbf{S}=small, \textbf{M}=medium, \textbf{L}=large) "
    r"and Wilcoxon signed-rank $p$-values corrected for multiple comparisons "
    r"via Benjamini--Hochberg (FDR) across all "
    + str(len(cell_keys)) + r" tests "
    r"($^{****}$: $p_{\mathrm{adj}}<0.0001$, $^{***}$: $p_{\mathrm{adj}}<0.001$, "
    r"$^{**}$: $p_{\mathrm{adj}}<0.01$, $^{*}$: $p_{\mathrm{adj}}<0.05$)."
)

PREAMBLE = "\n".join([
    r"% Add to LaTeX preamble:",
    r"% \usepackage[table]{xcolor}",
    r"% \definecolor{magnitudeS}{HTML}{FFF176}  % small  – light yellow",
    r"% \definecolor{magnitudeM}{HTML}{FFB74D}  % medium – orange",
    r"% \definecolor{magnitudeL}{HTML}{E57373}  % large  – red",
])

lines = [
    PREAMBLE,
    r"\begin{table*}[t]",
    r"\centering",
    rf"\caption{{{sig_note}}}",
    r"\label{tab:patch-structure}",
    r"\resizebox{\textwidth}{!}{%",
    rf"\begin{{tabular}}{{{col_spec}}}",
    r"\toprule",
    rf" & {model_headers} \\",
    cmidrule,
    rf"\textbf{{Metric}} & {sub_header} \\",
    r"\midrule",
    rf"\textit{{N (instances)}} & {' & '.join(n_cells)} \\",
    r"\midrule",
]

for display, key in METRIC_KEYS:
    cells = []
    for m in active_models:
        n, g_mean, l_mean, raw_p, mag = stat_grid[(key, m)]
        p_adj = adj_p_map[(key, m)]
        if n == 0:
            cells += [r"\text{--}", r"\text{--}", r"\text{--}"]
        else:
            cells += [fmt_val(l_mean), fmt_val(g_mean), fmt_sig(mag, p_adj)]
    lines.append(rf"{display} & {' & '.join(cells)} \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}}",
    r"\end{table*}",
]

out = OUT_DIR / "rq_table_patch_structure.tex"
out.write_text("\n".join(lines))
print(f"Saved → {out}")
print()

# ── Console summary ────────────────────────────────────────────────────────────
print(f"{'Metric':<28} {'Model':<10} {'Agent':>8} {'Human':>8}  Raw-p   Adj-p  Sig.")
print("-" * 78)
for display, key in METRIC_KEYS:
    for m in active_models:
        n, g_mean, l_mean, raw_p, mag = stat_grid[(key, m)]
        p_adj = adj_p_map[(key, m)]
        s = stars(p_adj)
        label = MODEL_LABELS[m]
        raw_str = f"{raw_p:.4f}" if not np.isnan(raw_p) else "  NaN "
        adj_str = f"{p_adj:.4f}" if not np.isnan(p_adj) else "  NaN "
        print(f"{display:<28} {label:<10} {l_mean:>8.3f} {g_mean:>8.3f}  {raw_str}  {adj_str}  {mag}{s}")
    print()
