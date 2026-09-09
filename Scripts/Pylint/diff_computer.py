"""
diff_computer.py — Compute before/after pylint issue count deltas per file.
"""
from pathlib import Path

import pandas as pd

from config import METRICS


class DiffComputer:
    """
    Joins before_<kind>.csv and after_<kind>.csv on the File column and
    computes per-metric deltas (after − before).

    Output columns:
        File, status, before_<M>, after_<M>, diff_<M>  for each M in METRICS
    """

    def compute(
        self,
        before_csv: Path,
        after_csv:  Path,
        diff_csv:   Path,
    ) -> None:
        print(f"  [diff] {before_csv.name} vs {after_csv.name}")

        b_rows = self._load(before_csv)
        a_rows = self._load(after_csv)

        b_dict = {r["File"]: r for r in b_rows}
        a_dict = {r["File"]: r for r in a_rows}

        diff_rows = []
        all_files = sorted(set(b_dict) | set(a_dict))

        for filepath in all_files:
            b = b_dict.get(filepath, {})
            a = a_dict.get(filepath, {})

            status = (
                "new"      if filepath not in b_dict else
                "removed"  if filepath not in a_dict else
                "modified"
            )

            row = {"File": filepath, "status": status}
            for m in METRICS:
                vb = float(b.get(m) or 0)
                va = float(a.get(m) or 0)
                row[f"before_{m}"] = vb
                row[f"after_{m}"]  = va
                row[f"diff_{m}"]   = va - vb
            diff_rows.append(row)

        cols = ["File", "status"] + [
            f"{prefix}_{m}" for m in METRICS for prefix in ("before", "after", "diff")
        ]
        diff_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diff_rows, columns=cols).to_csv(diff_csv, index=False)
        print(f"  [diff] {len(diff_rows)} files → {diff_csv.name}")

        if diff_rows:
            df   = pd.DataFrame(diff_rows)
            improved = (df["diff_n_total"] < 0).sum()
            worsened = (df["diff_n_total"] > 0).sum()
            neutral  = (df["diff_n_total"] == 0).sum()
            print(f"    n_total diff: improved={improved}  worsened={worsened}  neutral={neutral}")

    @staticmethod
    def _load(p: Path) -> list[dict]:
        return (
            pd.read_csv(p).to_dict(orient="records")
            if p.exists() and p.stat().st_size
            else []
        )
