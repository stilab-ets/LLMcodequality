"""
diff_computer.py — Compute before/after metric deltas per entity.
"""
from pathlib import Path

import pandas as pd

from config import METRICS


class DiffComputer:
    """
    Computes metric deltas (after − before) between two metrics CSVs.
    Each output row contains before_<M>, after_<M>, and diff_<M> for every metric.
    """

    def compute(
        self,
        before_csv: Path,
        after_csv: Path,
        diff_csv: Path,
    ) -> None:
        print(f"  [diff] {before_csv.name} vs {after_csv.name}")

        b_rows = self._load(before_csv)
        a_rows = self._load(after_csv)
        b_dict = {(r.get("Entity", ""), r.get("Kind", "")): r for r in b_rows}
        a_dict = {(r.get("Entity", ""), r.get("Kind", "")): r for r in a_rows}

        diff_rows = []
        for entity, kind in sorted(set(b_dict) | set(a_dict)):
            b = b_dict.get((entity, kind), {})
            a = a_dict.get((entity, kind), {})
            level  = b.get("Level") or a.get("Level")
            status = (
                "new"      if (entity, kind) not in b_dict else
                "removed"  if (entity, kind) not in a_dict else
                "modified"
            )
            row = {
                "Entity": entity,
                "Kind":   kind,
                "Level":  level,
                "status": status,
                "File":   b.get("File") or a.get("File"),
            }
            for m in METRICS:
                vb = float(b.get(m) or 0)
                va = float(a.get(m) or 0)
                row[f"before_{m}"] = vb
                row[f"after_{m}"]  = va
                row[f"diff_{m}"]   = va - vb
            diff_rows.append(row)

        cols = ["Entity", "Kind", "Level", "status", "File"] + [
            f"{prefix}_{m}" for m in METRICS for prefix in ("before", "after", "diff")
        ]
        diff_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diff_rows, columns=cols).to_csv(diff_csv, index=False)
        print(f"  [diff] {len(diff_rows)} rows → {diff_csv.name}")

        df = pd.DataFrame(diff_rows)
        if not df.empty:
            print(
                f"    File={df[df.Level == 'File'].shape[0]}  "
                f"Class={df[df.Level == 'Class'].shape[0]}  "
                f"Function={df[df.Level == 'Function'].shape[0]}"
            )

    @staticmethod
    def _load(p: Path) -> list[dict]:
        return (
            pd.read_csv(p).to_dict(orient="records")
            if p.exists() and p.stat().st_size
            else []
        )