"""
instance_symbol_values.py — For each instance directory in pylint_results,
read messages_diff_{variant}.csv and write symbol_values_{variant}.csv
alongside the existing files.

For symbols with numeric info in their message:
  status "removed" → before_value  (issue existed before the patch)
  status "new"     → after_value   (issue introduced by the patch)

Same (symbol, file, obj) pairs are merged into one row with before/after values.
Unmatched occurrences (only removed or only new) keep their own row.
"""
import re
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "pylint_results"


# ── extractors ────────────────────────────────────────────────────────────────

def _parens(msg: str) -> int | None:
    """Pull first number from '(actual/max)' pattern."""
    m = re.search(r"\((\d+)/\d+\)", msg)
    return int(m.group(1)) if m else None

def _try_stmts(msg: str) -> int | None:
    """Pull actual from 'contains N statements, expected at most M'."""
    m = re.search(r"contains\s+(\d+)\s+statements", msg)
    return int(m.group(1)) if m else None


SYMBOL_CONFIG: dict[str, tuple[str, callable]] = {
    "line-too-long":                ("long_line",             _parens),
    "too-few-public-methods":       ("few_public_methods",    _parens),
    "too-many-arguments":           ("many_arguments",        _parens),
    "too-many-boolean-expressions": ("many_boolean_expr",     _parens),
    "too-many-branches":            ("many_branches",         _parens),
    "too-many-instance-attributes": ("many_instance_attrs",   _parens),
    "too-many-lines":               ("many_lines",            _parens),
    "too-many-locals":              ("many_locals",           _parens),
    "too-many-nested-blocks":       ("many_nested_blocks",    _parens),
    "too-many-public-methods":      ("many_public_methods",   _parens),
    "too-many-return-statements":   ("many_return_stmts",     _parens),
    "too-many-statements":          ("many_statements",       _parens),
    "too-many-try-statements":      ("many_try_stmts",        _try_stmts),
    # too-many-function-args: no numeric info in message, skipped
}


def process_variant(instance_dir: Path, variant: str) -> None:
    src = instance_dir / f"messages_diff_{variant}.csv"
    out = instance_dir / f"symbol_values_{variant}.csv"

    if not src.exists():
        return

    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"  [WARN] {src}: {e}")
        return

    if "symbol" not in df.columns or "status" not in df.columns:
        return

    has_message = "message" in df.columns

    # (symbol, file, obj) -> {"removed": [(val, line), ...], "new": [(val, line), ...]}
    bucket: dict[tuple, dict[str, list]] = {}

    for _, row in df.iterrows():
        sym    = str(row.get("symbol", "")).strip()
        status = str(row.get("status", "")).strip().lower()

        if sym not in SYMBOL_CONFIG or status not in ("removed", "new"):
            continue

        _, extractor = SYMBOL_CONFIG[sym]
        msg = str(row.get("message", "")).strip() if has_message else ""
        val = extractor(msg)
        if val is None:
            continue

        key = (sym, str(row.get("file", "")).strip(), str(row.get("obj", "")).strip())
        if key not in bucket:
            bucket[key] = {"removed": [], "new": []}
        bucket[key][status].append(val)

    if not bucket:
        return

    rows = []
    for (sym, file_, obj_), sides in sorted(bucket.items()):
        removed_list = sides["removed"]
        new_list     = sides["new"]
        # pair removed↔new by position; leftovers become their own rows
        n = max(len(removed_list), len(new_list))
        for i in range(n):
            r  = removed_list[i] if i < len(removed_list) else None
            nw = new_list[i]     if i < len(new_list)     else None
            rows.append({
                "symbol":       sym,
                "metric":       SYMBOL_CONFIG[sym][0],
                "file":         file_,
                "obj":          obj_,
                "before_value": r,
                "after_value":  nw,
            })

    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    # collect all unique instance directories
    instance_dirs = sorted({
        p.parent
        for p in RESULTS_DIR.rglob("messages_diff_*.csv")
    })

    if not instance_dirs:
        print("No instance directories found under", RESULTS_DIR)
        sys.exit(1)

    print(f"Processing {len(instance_dirs)} instance directories...")

    written = 0
    for instance_dir in instance_dirs:
        for variant in ("llm", "gold"):
            out = instance_dir / f"symbol_values_{variant}.csv"
            process_variant(instance_dir, variant)
            if out.exists():
                written += 1

    print(f"Done. Wrote {written} symbol_values_*.csv files.")


if __name__ == "__main__":
    main()
