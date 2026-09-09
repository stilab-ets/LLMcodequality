"""
diff_computer.py — Compute metric diffs between before/after DPy outputs.
"""
from pathlib import Path

import pandas as pd

from config import (
    METRIC_KEY_COLS, METRIC_VALUE_COLS,
)


class DiffComputer:
    """
    Computes diffs between before and after DPy DataFrames.

    Smells  → set difference (added / removed rows), one row per unique smell instance.
    Metrics → numeric delta (after - before) per entity.
    """

    def diff_metrics(
        self,
        before:     pd.DataFrame,
        after:      pd.DataFrame,
        key_cols:   list[str],
        value_cols: list[str],
    ) -> pd.DataFrame:
        """
        Outer merge before/after on key_cols.
        Compute before_<M>, after_<M>, diff_<M> for every metric column.
        """
        if before.empty and after.empty:
            return pd.DataFrame()

        def safe_load(df):
            cols = [c for c in key_cols + value_cols if c in df.columns]
            return df[cols].copy() if not df.empty else pd.DataFrame(columns=key_cols + value_cols)

        b = safe_load(before)
        a = safe_load(after)

        for col in value_cols:
            if col not in b.columns: b[col] = 0.0
            if col not in a.columns: a[col] = 0.0

        valid_keys = [c for c in key_cols if c in b.columns and c in a.columns]
        merged = pd.merge(b, a, on=valid_keys, how="outer", suffixes=("_before", "_after"))

        for col in value_cols:
            merged[f"before_{col}"] = merged.get(f"{col}_before", merged.get(col, 0)).fillna(0)
            merged[f"after_{col}"]  = merged.get(f"{col}_after",  merged.get(col, 0)).fillna(0)
            merged[f"diff_{col}"]   = merged[f"after_{col}"] - merged[f"before_{col}"]
            merged.drop(
                columns=[c for c in [f"{col}_before", f"{col}_after"] if c in merged.columns],
                inplace=True,
            )

        out_cols = valid_keys + [
            f"{prefix}_{col}"
            for col in value_cols
            for prefix in ("before", "after", "diff")
            if f"{prefix}_{col}" in merged.columns
        ]
        return merged[out_cols]

    def compute(
        self,
        before_dfs: dict[str, pd.DataFrame],
        after_dfs:  dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Compute diffs for all DPy output files.
        Returns a dict keyed by file suffix.
        """
        diffs = {}

        for fname, key_cols in METRIC_KEY_COLS.items():
            b          = before_dfs.get(fname, pd.DataFrame())
            a          = after_dfs.get(fname,  pd.DataFrame())
            value_cols = METRIC_VALUE_COLS[fname]
            diffs[fname] = self.diff_metrics(b, a, key_cols, value_cols)
            print(f"  [metric diff] {fname}: {len(diffs[fname])} rows")

        return diffs

    def save(
        self,
        diffs:      dict[str, pd.DataFrame],
        before_dfs: dict[str, pd.DataFrame],
        after_dfs:  dict[str, pd.DataFrame],
        out_dir:    Path,
        kind:       str,
    ) -> None:
        """Save before, after, and diff CSVs with appropriate prefixes."""
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname, df in before_dfs.items():
            stem = Path(fname).stem
            df.to_csv(out_dir / f"before_{kind}_{stem}.csv", index=False, encoding="utf-8")

        for fname, df in after_dfs.items():
            stem = Path(fname).stem
            df.to_csv(out_dir / f"after_{kind}_{stem}.csv", index=False, encoding="utf-8")

        for fname, df in diffs.items():
            stem     = Path(fname).stem
            out_path = out_dir / f"diff_{kind}_{stem}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  [saved] {out_path.name}: {len(df)} rows")