"""
config.py — Global constants, paths, and DPy schema definitions.
"""
from pathlib import Path

# ── DPy executable ────────────────────────────────────────────────────────────
DPY_EXE = r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\Scripts\DPY\DPy.exe"

# ── Directories ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\repos")
MODEL     = "claude-45sonnet-10132025"
EVAL_DIR  = Path(r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\python_data\succeeded") / MODEL
OUT_DIR   = Path(f"dpy_results/{MODEL}")

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET_INSTANCE = ""   # pin to one instance ID for testing; "" = all
TARGET_REPO     = ""   # pin to one repo; "" = all

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

# ── DPy output file suffixes (DPy prefixes with input folder name) ────────────
DPY_METRIC_FILES = [
    "function_metrics.csv",
    "class_module_metrics.csv",
]
DPY_ALL_FILES = DPY_METRIC_FILES

# Column used to match rows to source files
DPY_FILE_COLUMN = {
    "function_metrics.csv":      "Module",
    "class_module_metrics.csv":  "Module",
}

# Key columns for metric diff (numeric delta)
METRIC_KEY_COLS = {
    "function_metrics.csv":     ["Project", "Package", "Module", "Class", "Method"],
    "class_module_metrics.csv": ["Project", "Package", "Module", "Class"],
}

METRIC_VALUE_COLS = {
    "function_metrics.csv":     ["LOC", "CC", "PC"],
    "class_module_metrics.csv": ["LOC", "WMC", "NOM", "NOPM", "NOF", "NOPF",
                                  "LCOM", "Fan-In", "Fan-Out", "DIT"],
}

# ── Pipeline flags ────────────────────────────────────────────────────────────
FILTER_PATCHES = True