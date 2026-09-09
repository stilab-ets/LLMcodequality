"""
metrics_aggregator.py — Aggregate per-instance diff CSVs into one flat CSV.
"""
from pathlib import Path

import pandas as pd

from config import METRICS, CLASS_METRICS, FUNCTION_METRICS, MAX_METRICS, MEAN_METRICS, OUT_DIR


class MetricsAggregator:
    """
    Aggregates diff CSVs across all instances into one flat CSV with
    per-level columns for gold and LLM patches.

    Column naming:
        diff_<Metric>_file_human     diff_<Metric>_file_llm
        diff_<Metric>_class_human    diff_<Metric>_class_llm
        diff_<Metric>_function_human diff_<Metric>_function_llm
    """

    def _agg_series(self, series: pd.Series, metric: str) -> float:
        s = series.dropna()
        if s.empty:
            return 0.0
        if metric in MAX_METRICS:
            return float(s.max())
        if metric in MEAN_METRICS:
            return float(s.mean())
        return float(s.sum())

    def aggregate_instance(self, diff_csv_path: Path, is_gold: bool = False) -> dict:
        """
        Aggregate diff metrics by File, Class, and Function levels for one instance.
        """
        suffix = "human" if is_gold else "llm"
        result = {
            f"diff_{m}_{lvl}_{suffix}": 0.0
            for m in METRICS
            for lvl in ("file", "class", "function")
        }

        if not diff_csv_path.exists() or diff_csv_path.stat().st_size == 0:
            return result

        df = pd.read_csv(diff_csv_path)
        masks = {
            "file":     df["Level"] == "File",
            "class":    df["Level"] == "Class",
            "function": df["Level"] == "Function",
        }

        for m in METRICS:
            col = f"diff_{m}"
            if col not in df.columns:
                continue
            lvl = (
                "class"    if m in CLASS_METRICS    else
                "function" if m in FUNCTION_METRICS else
                "file"
            )
            result[f"{col}_{lvl}_{suffix}"] = self._agg_series(
                df.loc[masks[lvl], col], m
            )

        return result

    def aggregate_all(
        self,
        valid_instances: list[dict],
        out_dir: Path = OUT_DIR,
    ) -> Path | None:
        """
        Aggregate all processed instances and save to final_aggregated_metrics.csv.
        Returns the output path or None if there was nothing to aggregate.
        """
        print("Aggregating metrics...")
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