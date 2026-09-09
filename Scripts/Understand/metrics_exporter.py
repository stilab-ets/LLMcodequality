"""
metrics_exporter.py — Run `und metrics` and filter the CSV to changed files.
"""
import csv
from pathlib import Path

import pandas as pd

from config import METRICS, LEVEL_KEYWORDS
from und_runner import UndRunner


class MetricsExporter:
    """
    Runs `und metrics` on a UDB, then filters the resulting CSV to keep
    only File, Class, and Function entities that belong to changed files.
    """

    def __init__(self, runner: UndRunner = None):
        self.runner = runner or UndRunner()

    def export(
        self,
        udb_path: Path,
        changed_files: list[str],
        out_csv: Path,
        miss_csv: Path | None = None,
        label: str = "",
    ) -> None:
        """
        Run `und metrics` on udb_path, filter to changed_files, save to out_csv.

        Matching strategy (endswith on normalized paths):
          - File entities:             Name column = relative path
          - Class/Function entities:   File column = declaring file (may be absolute)
          - Entity_Uniquename:         also checked as a fallback

        If miss_csv is given, a row per changed file is appended with status
        "found" or "missed" (missed = no entity matched for that file, usually
        because the file moved between the base-UDB commit and this instance).
        """
        print(f"  [export_metrics] running und metrics on {udb_path.name}...")
        self.runner.run(["metrics", str(udb_path)], timeout=900)

        metrics_csv = udb_path.with_suffix(".csv")
        if not metrics_csv.exists():
            print(f"  [export_metrics] WARNING: no CSV produced at {metrics_csv}")
            self._write_empty(out_csv)
            self._save_miss_stats(
                changed_files, matched_tails=set(),
                miss_csv=miss_csv, label=label,
            )
            return

        with open(metrics_csv, encoding="utf-8", errors="replace") as f:
            all_rows = list(csv.DictReader(f))
        print(f"  [export_metrics] {len(all_rows)} total rows in UDB metrics CSV")

        # Normalize changed-file paths for endswith matching
        tail_set = {p.replace("\\", "/") for p in changed_files}

        rows: list[dict] = []
        matched_tails: set[str] = set()

        for r in all_rows:
            kind       = r.get("Kind", "")
            name       = r.get("Name", "")
            decl_file  = r.get("File", "").replace("\\", "/")
            uniquename = r.get("Entity_Uniquename", "").replace("\\", "/")

            level = None
            for keyword, lvl in LEVEL_KEYWORDS:
                if keyword in kind:
                    level = lvl
                    break
            if level is None:
                continue

            file_path = name.replace("\\", "/") if level == "File" else decl_file

            # Match by uniquename substring first, then by file path endswith.
            # Also record which tail path triggered the match.
            hit = next(
                (t for t in tail_set if t in uniquename),
                None,
            )
            if hit is None and file_path:
                hit = next(
                    (
                        t for t in tail_set
                        if file_path == t
                        or file_path.endswith("/" + t)
                        or file_path.endswith(t)
                    ),
                    None,
                )
            if hit is None:
                continue

            matched_tails.add(hit)
            row = {"Entity": name, "Kind": kind, "Level": level, "File": file_path,
                   "_tail": hit}
            for m in METRICS:
                row[m] = r.get(m, "")
            rows.append(row)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        all_cols = ["Entity", "Kind", "Level", "File", "_tail"] + METRICS
        df_all   = pd.DataFrame(rows, columns=all_cols)

        # File-level entities use the full path as their name, so two paths
        # that resolve to the same changed file (different source roots in the
        # UDB) produce different entity names but the same matched tail.
        # Deduplicate by tail for File rows, by (Entity, Kind) for the rest.
        file_mask   = df_all["Level"] == "File"
        dedup_file  = df_all[file_mask].drop_duplicates(subset=["_tail"])
        dedup_other = df_all[~file_mask].drop_duplicates(subset=["Entity", "Kind"])
        df_out = (
            pd.concat([dedup_file, dedup_other], ignore_index=True)
            .drop(columns=["_tail"])
        )
        df_out.to_csv(out_csv, index=False)
        print(f"  [export_metrics] {len(rows)} matched rows ({len(df_out)} after dedup) → {out_csv.name}")
        print(
            f"    File={(df_out['Level'] == 'File').sum()}  "
            f"Class={(df_out['Level'] == 'Class').sum()}  "
            f"Function={(df_out['Level'] == 'Function').sum()}"
        )
        self._save_miss_stats(changed_files, matched_tails, miss_csv, label)

    def _save_miss_stats(
        self,
        changed_files: list[str],
        matched_tails: set[str],
        miss_csv: Path | None,
        label: str,
    ) -> None:
        """Log and optionally save per-file UDB match status."""
        tail_set = {p.replace("\\", "/") for p in changed_files}
        missed = sorted(tail_set - matched_tails)
        found  = sorted(matched_tails)

        tag = f"[{label}] " if label else ""
        print(
            f"  {tag}udb match: {len(found)}/{len(tail_set)} files found in UDB"
            + (f", {len(missed)} missed" if missed else "")
        )
        for f in missed:
            print(f"    ✗ udb-miss: {f}")

        if miss_csv is None:
            return

        rows = (
            [{"label": label, "file": f, "status": "found"}  for f in found] +
            [{"label": label, "file": f, "status": "missed"} for f in missed]
        )
        miss_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not miss_csv.exists()
        with open(miss_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["label", "file", "status"])
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"  {tag}udb miss stats saved → {miss_csv}")

    def _write_empty(self, out_csv: Path) -> None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["Entity", "Kind", "Level", "File"] + METRICS).to_csv(
            out_csv, index=False
        )