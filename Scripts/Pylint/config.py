"""
config.py — Global constants, paths, and Pylint metric definitions.
"""
from pathlib import Path

# ── Directories ─────────────────────────────────────────────────────────────
REPO_ROOT      = Path("/home/AV00500/Issam/SWE-Bench_Pro/repos")
SUCCEEDED_DIR  = Path("/home/AV00500/Issam/SWE-Bench_Pro/swebench_pro_pipeline/python_data/succeeded")

# Defaults used when running a single model (overridden per-model in main.py)
MODEL    = "claude-45haiku-10222025"
EVAL_DIR = SUCCEEDED_DIR / MODEL
OUT_DIR  = Path(f"pylint_results/{MODEL}")

# ── Dataset ──────────────────────────────────────────────────────────────────
TARGET_INSTANCE = ""   # pin to one instance ID for debugging; "" = all
TARGET_REPO     = ""   # pin to one repo e.g. "ansible__ansible"; "" = all

REPO_MAP = {
    "ansible__ansible":             "ansible",
    "qutebrowser__qutebrowser":     "qutebrowser",
    "internetarchive__openlibrary": "openlibrary",
}
REPO_LANG = {
    "ansible__ansible":             "Python",
    "qutebrowser__qutebrowser":     "Python",
    "internetarchive__openlibrary": "Python",
}
LANG_EXT = {
    "Python":     {".py"},
    "TypeScript": {".ts", ".tsx", ".js", ".jsx"},
    "Go":         {".go"},
}

# ── Pylint message categories ─────────────────────────────────────────────────
#   C = convention, R = refactor, W = warning, E = error, F = fatal
PYLINT_TYPES = ["convention", "refactor", "warning", "error", "fatal"]

# Metrics extracted per file: counts per category + totals
# These are summed across all files touched by a patch (instance-level aggregation).
METRICS = [
    "n_total",
    "n_convention",
    "n_refactor",
    "n_warning",
    "n_error",
    "n_fatal",
    "n_unique_symbols",
]

# ── Pipeline flags ────────────────────────────────────────────────────────────
FILTER_PATCHES = True   # strip test/doc/CI files from patches before applying
TEST_MODE      = False  # True = process only first instance per repo
N_WORKERS      = 3      # one worker per repo — should equal len(REPO_MAP)

# ── Pylint CLI options ────────────────────────────────────────────────────────
# Extra pylint flags passed on every invocation.
# --score=n   → suppress the "Your code has been rated at X/10" text
# --reports=n → suppress the full summary report
# --recursive=n → don't recursively analyze directories (we pass files explicitly)
PYLINT_EXTRA_ARGS = [
    "--score=n",
    "--reports=n",
    "--output-format=json",
]
