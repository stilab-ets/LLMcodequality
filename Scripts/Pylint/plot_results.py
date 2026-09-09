"""
plot_results.py — Visualize Pylint pipeline results.

Produces 7 figures covering aggregate diffs, per-category breakdowns,
win/tie/lose analysis, human-vs-LLM scatter, and top symbol charts.

Usage:
    python plot_results.py                          # all models in pylint_results/
    python plot_results.py --model claude-45haiku-10222025
    python plot_results.py --results-dir /path/to/pylint_results
    python plot_results.py --out plots/             # custom output directory
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

COLOR_HUMAN = "#2E86AB"   # blue
COLOR_LLM   = "#E84855"   # red
COLOR_NEW     = "#E84855"   # red  – messages introduced by patch
COLOR_REMOVED = "#3BB273"   # green – messages fixed by patch

CATEGORIES = ["convention", "refactor", "warning", "error", "fatal"]
CAT_COLORS  = ["#7B9E87", "#5B8DB8", "#F4A261", "#E76F51", "#C77DFF"]

SHORT_MODEL = {
    "claude-45haiku-10222025":   "Haiku",
    "claude-45sonnet-10132025":  "Sonnet",
    "codex":                     "Codex",
    "gemini-2-5-pro-nov17":      "Gemini",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_aggregated(results_dir: Path, model_filter: str | None = None) -> pd.DataFrame:
    """Load final_aggregated_metrics.csv for every model and concatenate."""
    frames = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name != model_filter:
            continue
        csv = model_dir / "final_aggregated_metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["model"] = model_dir.name
        df["model_short"] = SHORT_MODEL.get(model_dir.name, model_dir.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No final_aggregated_metrics.csv found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_all_messages(
    results_dir: Path,
    kind: str,                   # "gold" or "llm"
    model_filter: str | None = None,
) -> pd.DataFrame:
    """Load all messages_diff_{kind}.csv files and concatenate."""
    frames = []
    pattern = f"messages_diff_{kind}.csv"
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_filter and model_dir.name != model_filter:
            continue
        for csv in model_dir.rglob(pattern):
            if csv.stat().st_size == 0:
                continue
            df = pd.read_csv(csv)
            if df.empty:
                continue
            df["model"] = model_dir.name
            df["model_short"] = SHORT_MODEL.get(model_dir.name, model_dir.name)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["file", "status", "type", "message-id",
                                     "symbol", "line", "column", "obj", "message",
                                     "model", "model_short"])
    return pd.concat(frames, ignore_index=True)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path}")


def _model_palette(models: list[str]) -> dict[str, str]:
    fixed = {
        "Haiku":  COLOR_HUMAN,
        "Sonnet": COLOR_LLM,
        "Codex":  "#F4A261",
        "Gemini": "#3BB273",
    }
    palette = sns.color_palette("tab10", len(models))
    return {m: fixed.get(m, palette[i]) for i, m in enumerate(models)}


# ── Figure 1: Distribution of total diff (human vs LLM) ──────────────────────

def plot_total_diff_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Histogram + KDE of diff_n_total for human (gold) and LLM patches.
    One subplot per model if multiple models present.
    """
    models = df["model_short"].unique()
    ncols  = min(len(models), 3)
    nrows  = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, model in enumerate(sorted(models)):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df[df["model_short"] == model]

        for col, label, color in (
            ("diff_n_total_human", "Human (gold)", COLOR_HUMAN),
            ("diff_n_total_llm",   "LLM",          COLOR_LLM),
        ):
            vals = sub[col].dropna()
            ax.hist(vals, bins=30, alpha=0.45, color=color, density=True, label=label)
            vals.plot.kde(ax=ax, color=color, linewidth=2)

        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"{model}  (n={len(sub)})")
        ax.set_xlabel("Δ total issues  (negative = improved)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # hide unused axes
    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Distribution of Total Pylint Issue Change per Patch", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_total_diff_distribution.png")


# ── Figure 2: Category breakdown (mean diff per type) ────────────────────────

def plot_category_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bar chart: mean diff per pylint category for human vs LLM,
    one group of bars per model.
    """
    models = sorted(df["model_short"].unique())
    x      = np.arange(len(CATEGORIES))
    width  = 0.35 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = _model_palette(models)
    for m_idx, model in enumerate(models):
        sub   = df[df["model_short"] == model]
        human_means = [sub[f"diff_n_{cat}_human"].mean() for cat in CATEGORIES]
        llm_means   = [sub[f"diff_n_{cat}_llm"].mean()   for cat in CATEGORIES]

        offset = (m_idx - len(models) / 2 + 0.5) * width * 2.2
        col    = palette[model]

        bars_h = ax.bar(x + offset - width / 2, human_means, width,
                        color=col, alpha=0.9,   label=f"{model} human")
        bars_l = ax.bar(x + offset + width / 2, llm_means,   width,
                        color=col, alpha=0.45, label=f"{model} LLM",
                        hatch="///", edgecolor=col)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES])
    ax.set_xlabel("Pylint message category")
    ax.set_ylabel("Mean Δ issues  (negative = improved)")
    ax.set_title("Mean Issue Count Change per Category — Human vs LLM Patches")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_category_breakdown.png")


# ── Figure 3: Win / Tie / Lose (per model, human vs LLM) ─────────────────────

def plot_win_tie_lose(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal stacked bar: % of instances improved / neutral / worsened
    (based on diff_n_total < 0 / == 0 / > 0), split by model and patch kind.
    """
    records = []
    for model in sorted(df["model_short"].unique()):
        sub = df[df["model_short"] == model]
        for kind, col in (("Human", "diff_n_total_human"), ("LLM", "diff_n_total_llm")):
            vals = sub[col].dropna()
            n    = len(vals)
            records.append({
                "label":     f"{model} / {kind}",
                "improved":  (vals < 0).sum() / n * 100,
                "neutral":   (vals == 0).sum() / n * 100,
                "worsened":  (vals > 0).sum() / n * 100,
                "n":         n,
            })

    rdf = pd.DataFrame(records)
    labels = rdf["label"].tolist()
    y      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(labels) + 1.5)))

    bars_imp = ax.barh(y, rdf["improved"], color=COLOR_REMOVED, label="Improved (↓)")
    bars_neu = ax.barh(y, rdf["neutral"],  left=rdf["improved"],
                       color="#BBBBBB", label="Neutral (=)")
    bars_wor = ax.barh(y, rdf["worsened"], left=rdf["improved"] + rdf["neutral"],
                       color=COLOR_NEW, label="Worsened (↑)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("% of instances")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title("Win / Tie / Lose on Total Pylint Issue Count")
    ax.legend(loc="lower right", fontsize=9)

    for i, row in rdf.iterrows():
        ax.text(row["improved"] / 2, i, f"{row['improved']:.0f}%",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        if row["worsened"] > 5:
            ax.text(row["improved"] + row["neutral"] + row["worsened"] / 2, i,
                    f"{row['worsened']:.0f}%",
                    ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir, "fig3_win_tie_lose.png")


# ── Figure 4: Human vs LLM scatter ───────────────────────────────────────────

def plot_human_vs_llm_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter plot: diff_n_total_human vs diff_n_total_llm, one point per instance.
    Diagonal = equal quality; coloured by model.
    """
    models  = sorted(df["model_short"].unique())
    palette = _model_palette(models)

    fig, ax = plt.subplots(figsize=(7, 6))

    for model in models:
        sub = df[df["model_short"] == model]
        ax.scatter(
            sub["diff_n_total_human"], sub["diff_n_total_llm"],
            label=model, color=palette[model],
            alpha=0.55, s=30, linewidths=0,
        )

    lim_vals = pd.concat([df["diff_n_total_human"], df["diff_n_total_llm"]]).dropna()
    lo, hi   = lim_vals.min() - 2, lim_vals.max() + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Δ total issues — Human (gold) patch")
    ax.set_ylabel("Δ total issues — LLM patch")
    ax.set_title("Human vs LLM Patch: Total Pylint Issue Change per Instance")
    ax.legend(fontsize=9)

    # Quadrant labels
    margin = (hi - lo) * 0.03
    for txt, x, y in (
        ("Both improved", lo + margin, hi - margin * 4),
        ("Both worsened", hi - margin * 8, lo + margin),
        ("Only LLM better", lo + margin, lo + margin),
        ("Only Human better", hi - margin * 9, hi - margin * 4),
    ):
        ax.text(x, y, txt, fontsize=7, color="grey", alpha=0.8)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_human_vs_llm_scatter.png")


# ── Figure 5: Top symbols (removed vs new) ────────────────────────────────────

def plot_top_symbols(
    msgs_gold: pd.DataFrame,
    msgs_llm:  pd.DataFrame,
    out_dir:   Path,
    n: int = 15,
) -> None:
    """
    Two rows × two columns:
      - Top N removed symbols for gold / LLM  (green → patch fixed these)
      - Top N new symbols for gold / LLM      (red   → patch introduced these)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Top {n} Most Frequent Pylint Symbols — Removed vs New", fontsize=13)

    specs = [
        (axes[0, 0], msgs_gold, "removed", f"Human patch — Removed (fixed)  [{COLOR_REMOVED}]", COLOR_REMOVED),
        (axes[0, 1], msgs_llm,  "removed", f"LLM patch   — Removed (fixed)",                     COLOR_REMOVED),
        (axes[1, 0], msgs_gold, "new",     f"Human patch — New (introduced)",                     COLOR_NEW),
        (axes[1, 1], msgs_llm,  "new",     f"LLM patch   — New (introduced)",                     COLOR_NEW),
    ]

    for ax, msgs, status, title, color in specs:
        sub = msgs[msgs["status"] == status]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue
        counts = (
            sub.groupby("symbol")
               .size()
               .sort_values(ascending=True)
               .tail(n)
        )
        bars = ax.barh(counts.index, counts.values, color=color, alpha=0.8)
        ax.set_xlabel("Count across all instances")
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig5_top_symbols.png")


# ── Figure 6: Message type breakdown (new vs removed, gold vs LLM) ───────────

def plot_message_type_breakdown(
    msgs_gold: pd.DataFrame,
    msgs_llm:  pd.DataFrame,
    out_dir:   Path,
) -> None:
    """
    Grouped bar chart: for each (patch kind × status) combo, how many messages
    of each pylint type were new vs removed.
    """
    type_order = ["fatal", "error", "warning", "refactor", "convention"]

    records = []
    for msgs, patch in ((msgs_gold, "Human"), (msgs_llm, "LLM")):
        for status in ("removed", "new"):
            sub = msgs[msgs["status"] == status]
            for t in type_order:
                records.append({
                    "patch":  patch,
                    "status": status,
                    "type":   t,
                    "count":  (sub["type"] == t).sum(),
                })

    rdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Pylint Message Type Breakdown — New vs Removed by Patch", fontsize=13)

    type_colors = dict(zip(type_order, ["#C77DFF", "#E76F51", "#F4A261", "#5B8DB8", "#7B9E87"]))

    for ax, status, title in (
        (axes[0], "removed", "Removed (fixed by patch)"),
        (axes[1], "new",     "New (introduced by patch)"),
    ):
        sub = rdf[rdf["status"] == status]
        x   = np.arange(len(type_order))
        width = 0.35

        for p_idx, patch in enumerate(["Human", "LLM"]):
            vals = [
                sub[(sub["patch"] == patch) & (sub["type"] == t)]["count"].sum()
                for t in type_order
            ]
            offset = (p_idx - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                color=[type_colors[t] for t in type_order],
                alpha=0.9 if patch == "Human" else 0.5,
                edgecolor="white",
                label=patch,
            )
            ax.bar_label(bars, padding=1, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in type_order])
        ax.set_title(title)
        ax.set_ylabel("Message count")
        ax.legend(title="Patch", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig6_message_type_breakdown.png")


# ── Figure 7: Per-instance sorted diff ───────────────────────────────────────

def plot_per_instance_sorted(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Sorted bar chart of diff_n_total per instance (one subplot per model).
    Human and LLM bars are overlaid with transparency so both are visible.
    Instances sorted by human diff.
    """
    models = sorted(df["model_short"].unique())
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(max(14, len(df) // 4), 4.5 * len(models)),
                             squeeze=False)

    for idx, model in enumerate(models):
        ax  = axes[idx][0]
        sub = df[df["model_short"] == model].copy()
        sub = sub.sort_values("diff_n_total_human").reset_index(drop=True)
        x   = np.arange(len(sub))

        ax.bar(x, sub["diff_n_total_human"], color=COLOR_HUMAN, alpha=0.75,
               label="Human", width=1.0)
        ax.bar(x, sub["diff_n_total_llm"], color=COLOR_LLM, alpha=0.55,
               label="LLM", width=1.0)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{model}  —  Δ total issues per instance (sorted by human diff)")
        ax.set_xlabel("Instance (sorted by human diff)")
        ax.set_ylabel("Δ total issues")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.5, len(sub) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig7_per_instance_sorted.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pylint pipeline results")
    parser.add_argument("--results-dir", default="pylint_results",
                        help="Root directory of pylint_results (default: pylint_results/)")
    parser.add_argument("--model", default=None,
                        help="Restrict to one model directory name")
    parser.add_argument("--out", default=None,
                        help="Output directory for plots (default: <results-dir>/plots/)")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of symbols to show in top-symbols chart (default: 15)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out) if args.out else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {results_dir} ...")
    df        = load_aggregated(results_dir, model_filter=args.model)
    msgs_gold = load_all_messages(results_dir, "gold", model_filter=args.model)
    msgs_llm  = load_all_messages(results_dir, "llm",  model_filter=args.model)

    models = df["model_short"].unique().tolist()
    print(f"  Models  : {models}")
    print(f"  Instances: {len(df)}")
    print(f"  Gold messages: {len(msgs_gold)}")
    print(f"  LLM  messages: {len(msgs_llm)}")
    print(f"  Output dir: {out_dir}\n")

    print("Generating figures...")
    plot_total_diff_distribution(df, out_dir)
    plot_category_breakdown(df, out_dir)
    plot_win_tie_lose(df, out_dir)
    plot_human_vs_llm_scatter(df, out_dir)
    plot_top_symbols(msgs_gold, msgs_llm, out_dir, n=args.top_n)
    plot_message_type_breakdown(msgs_gold, msgs_llm, out_dir)
    plot_per_instance_sorted(df, out_dir)

    print(f"\nAll done — {len(list(out_dir.glob('*.png')))} figures in {out_dir}")


if __name__ == "__main__":
    main()
