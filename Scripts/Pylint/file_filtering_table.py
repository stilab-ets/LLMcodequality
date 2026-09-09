#!/usr/bin/env python3
"""
file_filtering_table.py — File classification & filtering statistics per model.

Reproduces the file-dropping rules of ``patch_cleaner.PatchCleaner`` (the rules
applied before pylint analysis) and reports, for each model's *own* (LLM)
patches, how many changed files fall into each non-production category.

A file block is DROPPED by the cleaner when EITHER:
  * it matches an auxiliary pattern (test / docs / config / helper-script), OR
  * it is a newly-added file at the repository root.

We reconstruct the *reason* for each drop from the recorded
``patch_filter_stats.csv`` files. Because the only two drop reasons are the two
above, a dropped file that matches no auxiliary pattern was necessarily dropped
as a root-level addition. Categories are mutually exclusive (priority order:
Test > Docs > Config > Auxiliary > Root-level), so the per-category counts sum
exactly to the Dropped total.

Output → pylint_results/results/file_filtering_table.tex
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

HERE       = Path(__file__).parent
RESULTS    = HERE / "pylint_results"
OUT_DIR    = RESULTS / "results"
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
    "claude-45haiku-10222025":  "Claude Haiku",
    "claude-45sonnet-10132025": "Claude Sonnet",
    "codex":                    "Codex",
    "gemini-2-5-pro-nov17":     "Gemini",
    "glm-4p5-10222025":         "GLM",
    "gpt-5-high-paper":         "GPT-5 High",
    "gptoss-paper":             "GPT-OSS",
}

# ── Category patterns (mutually exclusive, evaluated in this order) ─────────────
# Mirrors patch_cleaner.PatchCleaner.AUXILIARY_PATTERNS, grouped by intent.
TEST_PATTERNS = [
    r"(^|/)tests?/",
    r"(^|/)testing/",
    r"(^|/)__tests?__/",
    r"(^|/)testdata/",
    r"(^|/)conftest\.py$",
    # test modules in any language the benchmark repos use, not just Python
    r"(^|/)test_[^/]+\.(py|js|jsx|ts|tsx|html|go)$",
    r"(^|/)[^/]+_tests?\.(py|js|jsx|ts|tsx|go)$",
    r"(^|/)[^/]+\.(test|spec)\.(js|jsx|ts|tsx)$",
    r"(^|/)tests\.py$",
    r"(^|/)test\.py$",
    r"(^|/)test_implementation\.py$",
    r"^quick_test\.py$",
    # NOTE: root-anchored on purpose — a nested `validation.py` is production
    # code (e.g. lib/ansible/module_utils/common/validation.py). Keep in sync
    # with patch_cleaner.PatchCleaner.AUXILIARY_PATTERNS.
    r"^final_validation\.py$",
    r"^validation\.py$",
]
DOCS_PATTERNS = [
    r"(^|/)(docs?|documentation)/",
    r"\.(md|rst|txt|adoc|asciidoc)$",
    r"(^|/)(README|CHANGELOG|CHANGES|CONTRIBUTING|HISTORY|AUTHORS|LICENSE|NOTICE)(\.[a-z]+)?$",
    r"^IMPLEMENTATION_SUMMARY\.md$",
    r"^[^/]+\.txt$",
    r"^notes\.md$",
    r"^TODO\.md$",
    r"^NOTES\.md$",
]
CONFIG_PATTERNS = [
    r"\.(yml|yaml|cfg|ini|conf|toml|json|xml)$",
    r"(^|/)(Makefile|Dockerfile|Jenkinsfile|docker-compose\.[^/]+)$",
    r"(^|/)\.github/",
    r"(^|/)\.circleci/",
    r"(^|/)\.travis\.yml$",
]
# Agent-generated helper / scratch scripts (demo, example, verify, …)
AUX_PATTERNS = [
    r"^demo_.*\.py$",
    r"^example_.*\.py$",
    r"^implement_.*\.py$",
    r"^setup_.*\.py$",
    r"^[^/]+_demo\.py$",
    r"^[^/]+_example\.py$",
    r"^[^/]+_implementation\.py$",
    r"^verify_.*\.py$",
    r"^validation_.*\.py$",
    r"^check_.*\.py$",
    r"^run_.*\.py$",
    r"^usage_.*\.py$",
]

CATEGORIES = [
    ("Test",   [re.compile(p, re.IGNORECASE) for p in TEST_PATTERNS]),
    ("Docs",   [re.compile(p, re.IGNORECASE) for p in DOCS_PATTERNS]),
    ("Config", [re.compile(p, re.IGNORECASE) for p in CONFIG_PATTERNS]),
    ("Aux",    [re.compile(p, re.IGNORECASE) for p in AUX_PATTERNS]),
]


def normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("ab/")


def classify_dropped(path: str) -> str:
    """Assign a dropped file to exactly one category (priority-ordered).

    Any dropped file not matching an auxiliary pattern was dropped because it
    is a newly-added root-level file → 'Root'.
    """
    p = normalize(path)
    for name, regexes in CATEGORIES:
        if any(r.search(p) for r in regexes):
            return name
    return "Root"


# ── Tally over all recorded llm patch-filter stats ─────────────────────────────
CAT_ORDER = ["Test", "Docs", "Config", "Aux", "Root"]
counts = {m: defaultdict(int) for m in MODELS}   # cat -> n
totals = {m: 0 for m in MODELS}                  # all llm files (kept+dropped)
dropped_tot = {m: 0 for m in MODELS}

for m in MODELS:
    model_dir = RESULTS / m
    if not model_dir.exists():
        continue
    for stats_csv in model_dir.rglob("patch_filter_stats.csv"):
        with open(stats_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("patch") != "llm":
                    continue
                totals[m] += 1
                if row["status"] == "dropped":
                    dropped_tot[m] += 1
                    counts[m][classify_dropped(row["file"])] += 1

# ── Emit LaTeX ─────────────────────────────────────────────────────────────────
# Display columns: (header, [internal categories summed into it])
# 'Aux' (named helper scripts) and 'Root' (stray root-level additions) are the
# same phenomenon — agent-generated throwaway files — so they are merged.
DISPLAY_COLS = [
    ("Test",      ["Test"]),
    ("Docs",      ["Docs"]),
    ("Config",    ["Config"]),
    ("Non-source", ["Aux", "Root"]),
]


def col_count(c, cats):
    return sum(c[k] for k in cats)

# Codex and GPT-5 Codex submit identical patches → show one (Codex) row.
DROP_DUPLICATE = {"gpt-5-codex-debug-oct22"}
active = [m for m in MODELS if totals[m] > 0 and m not in DROP_DUPLICATE]

lines = [
    r"\begin{table}[!t]",
    r"\centering",
    r"\caption{Classification of non-production files filtered from the "
    r"agent-submitted patches prior to static analysis. Counts are numbers of "
    r"changed files. The category columns are mutually exclusive and sum "
    r"to \textbf{Dropped}.}",
    r"\label{tab:file-filtering}",
    r"\begin{tabular}{@{}l c c c c c c c@{}}",
    r"\toprule",
    r"\textbf{Model} & " +
    " & ".join(rf"\textbf{{{h}}}" for h, _ in DISPLAY_COLS) +
    r" & \textbf{Dropped} & \textbf{Total} & \textbf{\% Drop} \\",
    r"\midrule",
]

sum_cat = defaultdict(int)
sum_drop = 0
sum_total = 0
for m in active:
    c = counts[m]
    drop = dropped_tot[m]
    tot = totals[m]
    pct = 100.0 * drop / tot if tot else 0.0
    cells = [str(col_count(c, cats)) for _, cats in DISPLAY_COLS]
    lines.append(
        f"{MODEL_LABELS[m]} & " + " & ".join(cells) +
        f" & {drop} & {tot} & {pct:.1f}\\% \\\\"
    )
    for k in CAT_ORDER:
        sum_cat[k] += c[k]
    sum_drop += drop
    sum_total += tot

pct_tot = 100.0 * sum_drop / sum_total if sum_total else 0.0
lines += [
    r"\midrule",
    r"\textbf{Total} & " +
    " & ".join(str(sum(sum_cat[k] for k in cats)) for _, cats in DISPLAY_COLS) +
    f" & {sum_drop} & {sum_total:,} & {pct_tot:.1f}\\% \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"",
    r"\vspace{2pt}",
    r"\begin{flushleft}\footnotesize",
    r"\textbf{Test}: test modules, test directories and test fixtures "
    r"(\texttt{test\_*}, \texttt{*\_test.*}, \texttt{*.spec.*}, "
    r"\texttt{conftest.py}, \texttt{tests/}, \texttt{testing/}). "
    r"\textbf{Docs}: documentation and prose "
    r"(\texttt{.md}, \texttt{.rst}, \texttt{.txt}, \texttt{.adoc}, "
    r"\texttt{docs/}). "
    r"\textbf{Config}: build and CI configuration "
    r"(\texttt{.yml}, \texttt{.yaml}, \texttt{.ini}, \texttt{.cfg}, "
    r"\texttt{.toml}, \texttt{.json}). "
    r"\textbf{Non-source}: agent-generated scratch files not part of the code base "
    r"--- named helper scripts (\texttt{demo\_*.py}, \texttt{verify\_*.py}, "
    r"\texttt{check\_*.py}, \texttt{run\_*.py}, \texttt{*\_implementation.py}) "
    r"and other newly-created files added at the repository root "
    r"(predominantly \texttt{debug\_*.py} probes). "
    r"Categories are evaluated in priority order, so each filtered "
    r"file is counted once. Codex and GPT-5 Codex submitted identical patches; "
    r"a single Codex row is reported.",
    r"\end{flushleft}",
    r"\end{table}",
]

tex = "\n".join(lines)
out = OUT_DIR / "file_filtering_table.tex"
out.write_text(tex)

# ── Console summary ────────────────────────────────────────────────────────────
hdr = ["Model", *[h for h, _ in DISPLAY_COLS], "Dropped", "Total", "%Drop"]
print(f"{hdr[0]:<14}" + "".join(f"{h:>10}" for h in hdr[1:]))
for m in active:
    c = counts[m]
    row = [str(col_count(c, cats)) for _, cats in DISPLAY_COLS]
    pct = 100.0 * dropped_tot[m] / totals[m]
    print(f"{MODEL_LABELS[m]:<14}" +
          "".join(f"{v:>10}" for v in row) +
          f"{dropped_tot[m]:>10}{totals[m]:>10}{pct:>9.1f}%")
print(f"{'Total':<14}" +
      "".join(f"{sum(sum_cat[k] for k in cats):>10}" for _, cats in DISPLAY_COLS) +
      f"{sum_drop:>10}{sum_total:>10}{pct_tot:>9.1f}%")
print(f"\nSaved → {out}")
