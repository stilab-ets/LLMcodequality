"""
config.py — Global constants, paths, and metric definitions.
"""
from pathlib import Path

# ── SciTools ────────────────────────────────────────────────────────────────
SCI_BIN = "/home/AV00500/HULLM/scitools/bin/linux64"
UND_EXE = f"{SCI_BIN}/und"

# ── Directories ─────────────────────────────────────────────────────────────
UDB_DIR        = Path("understand_dbs")
REPO_ROOT      = Path("/home/AV00500/Issam/SWE-Bench_Pro/repos")
SUCCEEDED_DIR  = Path("/home/AV00500/Issam/SWE-Bench_Pro/swebench_pro_pipeline/python_data/succeeded")

# Defaults used when running a single model (overridden per-model in main.py)
MODEL     = "claude-45haiku-10222025"
EVAL_DIR  = SUCCEEDED_DIR / MODEL
OUT_DIR   = Path(f"understand_results/{MODEL}")

# ── Dataset ─────────────────────────────────────────────────────────────────
TARGET_INSTANCE = ""   # pin to one instance ID for testing; "" = all
TARGET_REPO     = ""   # pin to one repo ID e.g. "ansible__ansible"; "" = all

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

# ── Understand UDB settings ──────────────────────────────────────────────────
UDB_SETTINGS = {
    "-MetricShowDeclaredInFile":                "On",
    "-MetricFileNameDisplayMode":               "RelativePath",
    "-MetricShowFunctionParameterTypes":        "On",
    "-MetricAddUniqueNameColumn":               "On",
    "-MetricCyclomatic":                        "All",
    "-MetricShowEssential":                     "On",
    "-MetricShowPaths":                         "On",
    "-MetricShowMaximumNesting":                "On",
    "-MetricShowCouplingAndCohesionMetrics":    "On",
    "-MetricShowInheritanceMetrics":            "On",
    "-MetricShowMethodAndVariableCountMetrics": "On",
    "-MetricShowLineCountMetrics":              "On",
    "-MetricShowStatementCountMetrics":         "On",
    "-MetricShowAggregatedFunctionMetrics":     "On",
    "-MetricShowAggregatedClassMetrics":        "On",
    "-MetricShowAggregatedFileMetrics":         "On",
}

# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS = [
    "AvgCountLine", "AvgCountLineBlank", "AvgCountLineCode", "AvgCountLineComment",
    "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgCyclomaticStrictModified",
    "AvgEssential", "CCViolDensityCode", "CCViolDensityLine", "CountCCViol",
    "CountCCViolType", "CountClassBase", "CountClassCoupled", "CountClassCoupledModified",
    "CountClassDerived", "CountDeclClass", "CountDeclClassMethod", "CountDeclClassVariable",
    "CountDeclExecutableUnit", "CountDeclFile", "CountDeclFunction", "CountDeclInstanceMethod",
    "CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodAll", "CountDeclMethodDefault",
    "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic", "CountInput",
    "CountLine", "CountLineBlank", "CountLineCode", "CountLineCodeDecl",
    "CountLineCodeExe", "CountLineComment", "CountOutput", "CountPath",
    "CountPathLog", "CountSemicolon", "CountStmt", "CountStmtDecl",
    "CountStmtExe", "Cyclomatic", "CyclomaticModified", "CyclomaticStrict",
    "CyclomaticStrictModified", "Essential", "MaxCyclomatic", "MaxCyclomaticModified",
    "MaxCyclomaticStrict", "MaxCyclomaticStrictModified", "MaxEssential", "MaxInheritanceTree",
    "MaxNesting", "PercentLackOfCohesion", "PercentLackOfCohesionModified", "RatioCommentToCode",
    "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumCyclomaticStrictModified",
    "SumEssential",
]

# Maps Understand Kind substrings → normalized Level (order matters)
LEVEL_KEYWORDS = [
    ("File",     "File"),
    ("Class",    "Class"),
    ("Function", "Function"),
    ("Method",   "Function"),
]

# ── Aggregation sets ─────────────────────────────────────────────────────────
CLASS_METRICS = {
    "CountClassBase", "CountClassCoupled", "CountClassCoupledModified",
    "CountClassDerived", "CountDeclInstanceMethod", "CountDeclInstanceVariable",
    "CountDeclMethod", "CountDeclMethodAll", "CountDeclMethodDefault",
    "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic",
    "MaxInheritanceTree", "PercentLackOfCohesion", "PercentLackOfCohesionModified",
}
FUNCTION_METRICS = {
    "Cyclomatic", "CyclomaticModified", "CyclomaticStrict", "CyclomaticStrictModified",
    "Essential", "MaxNesting", "CountInput", "CountOutput", "CountPath", "CountPathLog",
}
SUM_METRICS = {
    "CountCCViol", "CountCCViolType",
    "CountClassBase", "CountClassCoupled", "CountClassCoupledModified", "CountClassDerived",
    "CountDeclClass", "CountDeclClassMethod", "CountDeclClassVariable",
    "CountDeclExecutableUnit", "CountDeclFile", "CountDeclFunction",
    "CountDeclInstanceMethod", "CountDeclInstanceVariable",
    "CountDeclMethod", "CountDeclMethodAll", "CountDeclMethodDefault",
    "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic",
    "CountInput", "CountLine", "CountLineBlank", "CountLineCode",
    "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment",
    "CountOutput", "CountPath", "CountPathLog",
    "CountSemicolon", "CountStmt", "CountStmtDecl", "CountStmtExe",
    "SumCyclomatic", "SumCyclomaticModified",
    "SumCyclomaticStrict", "SumCyclomaticStrictModified", "SumEssential","MaxNesting",
}
MAX_METRICS = {
    "MaxCyclomatic", "MaxCyclomaticModified",
    "MaxCyclomaticStrict", "MaxCyclomaticStrictModified",
    "MaxEssential", "MaxInheritanceTree", 
}
MEAN_METRICS = {
    "AvgCountLine", "AvgCountLineBlank", "AvgCountLineCode", "AvgCountLineComment",
    "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgCyclomaticStrictModified",
    "AvgEssential",
    "CCViolDensityCode", "CCViolDensityLine",
    "Cyclomatic", "CyclomaticModified", "CyclomaticStrict", "CyclomaticStrictModified",
    "Essential",
    "PercentLackOfCohesion", "PercentLackOfCohesionModified",
    "RatioCommentToCode",
}

# ── Pipeline flags ───────────────────────────────────────────────────────────
FILTER_PATCHES = True   # set to False to apply patches as-is
TEST_MODE      = False  # True = process only first instance per repo