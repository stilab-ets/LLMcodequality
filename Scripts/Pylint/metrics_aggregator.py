"""
metrics_aggregator.py — Aggregate per-instance diff CSVs into one flat CSV.
"""
from pathlib import Path

import pandas as pd

from config import METRICS, OUT_DIR


class MetricsAggregator:
    """
    Reads every diff_gold.csv / diff_llm.csv produced by the pipeline and
    rolls them up into a single final_aggregated_metrics.csv with one row
    per instance.

    All METRICS are issue counts, so they are summed across files.
    Column naming: diff_<Metric>_human  /  diff_<Metric>_llm
    """

    def aggregate_instance(self, diff_csv_path: Path, is_gold: bool = False) -> dict:
        suffix = "human" if is_gold else "llm"
        result = {f"diff_{m}_{suffix}": 0.0 for m in METRICS}

        if not diff_csv_path.exists() or diff_csv_path.stat().st_size == 0:
            return result

        df = pd.read_csv(diff_csv_path)
        for m in METRICS:
            col = f"diff_{m}"
            if col in df.columns:
                result[f"diff_{m}_{suffix}"] = float(df[col].sum())

        return result

    def aggregate_all(
        self,
        valid_instances: list[dict],
        out_dir: Path = OUT_DIR,
    ) -> Path | None:
        print("Aggregating pylint metrics...")
        agg_rows = []

        for inst in valid_instances:
            inst_out = out_dir / inst["repo_name"] / inst["iid"]
            if not inst_out.exists():
                continue
            row = {"instance_id": inst["iid"], "repo": inst["repo_name"]}
            row.update(self.aggregate_instance(inst_out / "diff_gold.csv", is_gold=True))
            row.update(self.aggregate_instance(inst_out / "diff_llm.csv",  is_gold=False))
            agg_rows.append(row)

        if not agg_rows:
            print("  → no rows to aggregate yet")
            return None

        agg_path = out_dir / "final_aggregated_metrics.csv"
        pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
        print(f"  → {len(agg_rows)} rows aggregated → {agg_path}")
        return agg_path
