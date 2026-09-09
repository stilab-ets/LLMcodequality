"""
aggregate_metrics.py — Aggregate DPy diff results into one row per instance.

For metric files (function_metrics, class_module_metrics):
    sum all diff_* columns across rows.

Output: dpy_results/aggregated.csv
        one row per (model, repo, iid), with gold_ and llm_ prefixed columns.

Usage:
    python aggregate_metrics.py
"""
from pathlib import Path

import pandas as pd

DPY_RESULTS = Path(__file__).parent / "dpy_results"
OUT_CSV     = DPY_RESULTS / "aggregated.csv"

METRIC_FILES = ["function_metrics", "class_module_metrics"]
KINDS        = ["gold", "llm"]


def _sum_metrics(csv_path: Path) -> dict:
    """Return sum of every diff_* column in a metric diff CSV."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    return {c: float(df[c].sum()) for c in diff_cols}



def aggregate_instance(inst_dir: Path) -> dict:
    row = {}
    for kind in KINDS:
        for mfile in METRIC_FILES:
            prefix = "fn" if mfile == "function_metrics" else "cls"
            sums = _sum_metrics(inst_dir / f"diff_{kind}_{mfile}.csv")
            for col, val in sums.items():
                metric = col[len("diff_"):]
                row[f"{kind}_{prefix}_{metric}"] = val
    return row


def main():
    if not DPY_RESULTS.exists():
        print(f"dpy_results/ not found — nothing to aggregate.")
        return

    records = []

    for model_dir in sorted(DPY_RESULTS.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "_tmp":
            continue
        model = model_dir.name

        for repo_dir in sorted(model_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            repo = repo_dir.name

            for inst_dir in sorted(repo_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                iid = inst_dir.name

                row = aggregate_instance(inst_dir)
                row = {"model": model, "repo": repo, "iid": iid, **row}
                records.append(row)

    if not records:
        print("No processed instances found.")
        return

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_CSV}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
