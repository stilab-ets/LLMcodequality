"""
pylint_exporter.py — Run pylint on changed files and export per-file issue counts + raw messages.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import METRICS, PYLINT_TYPES
from pylint_runner import PylintRunner


class PylintExporter:
    """
    Runs pylint on a set of changed files (relative to repo root) and writes:

    1. <out_csv>  — per-file issue counts (File, n_total, n_convention, …)
    2. <out_csv>.messages.json — normalized raw message list for later diffing

    The JSON file holds a list of dicts with these fields:
        file, type, message-id, symbol, message, line, column, endLine, endColumn, obj
    where `file` is the normalized repo-relative path.
    """

    def __init__(self, runner: PylintRunner = None):
        self.runner = runner or PylintRunner()

    def export(
        self,
        repo_path: Path,
        changed_files: list[str],
        out_csv: Path,
    ) -> None:
        """
        Run pylint on *changed_files* (relative paths from *repo_path*) and save:
        - per-file count CSV
        - raw message JSON (same stem + '.messages.json')
        """
        messages_json_path = out_csv.with_suffix("").with_suffix(".messages.json")

        if not changed_files:
            self._write_empty(out_csv, changed_files, messages_json_path)
            return

        raw_messages = self.runner.run(changed_files, cwd=repo_path)

        repo_str  = str(repo_path).rstrip("/") + "/"
        file_data: dict[str, dict] = {f: self._zero_row(f) for f in changed_files}
        # normalized messages keyed by matched file
        file_messages: dict[str, list[dict]] = {f: [] for f in changed_files}

        for msg in raw_messages:
            msg_path = msg.get("path", "").replace("\\", "/")
            if msg_path.startswith(repo_str):
                rel = msg_path[len(repo_str):]
            else:
                rel = msg_path

            matched_key = next(
                (f for f in changed_files if rel == f or rel.endswith("/" + f) or f.endswith(rel)),
                None,
            )
            if matched_key is None:
                continue

            row    = file_data[matched_key]
            mtype  = msg.get("type", "").lower()
            symbol = msg.get("symbol", "")

            row["n_total"] += 1
            if mtype in PYLINT_TYPES:
                row[f"n_{mtype}"] += 1
            row["_symbols"].add(symbol)

            file_messages[matched_key].append({
                "file":       matched_key,
                "type":       mtype,
                "message-id": msg.get("message-id", ""),
                "symbol":     symbol,
                "message":    msg.get("message", ""),
                "line":       msg.get("line"),
                "column":     msg.get("column"),
                "endLine":    msg.get("endLine"),
                "endColumn":  msg.get("endColumn"),
                "obj":        msg.get("obj", ""),
            })

        rows = []
        for f, row in file_data.items():
            row["n_unique_symbols"] = len(row.pop("_symbols"))
            rows.append(row)

        # ── write count CSV ───────────────────────────────────────────────
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["File"] + METRICS).to_csv(out_csv, index=False)

        # ── write raw messages JSON ───────────────────────────────────────
        all_messages = [m for msgs in file_messages.values() for m in msgs]
        messages_json_path.write_text(
            json.dumps(all_messages, indent=2), encoding="utf-8"
        )

        total = sum(r["n_total"] for r in rows)
        print(
            f"  [pylint_export] {len(rows)} files, {total} total issues → {out_csv.name}"
        )
        self._print_breakdown(rows)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_row(filepath: str) -> dict:
        return {
            "File":             filepath,
            "n_total":          0,
            "n_convention":     0,
            "n_refactor":       0,
            "n_warning":        0,
            "n_error":          0,
            "n_fatal":          0,
            "n_unique_symbols": 0,
            "_symbols":         set(),
        }

    @staticmethod
    def _print_breakdown(rows: list[dict]) -> None:
        totals = defaultdict(int)
        for r in rows:
            for k in METRICS:
                totals[k] += r.get(k, 0)
        parts = "  ".join(f"{k}={totals[k]}" for k in METRICS)
        print(f"    {parts}")

    def _write_empty(
        self,
        out_csv: Path,
        changed_files: list[str],
        messages_json_path: Path,
    ) -> None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._zero_row(f) for f in changed_files]
        for r in rows:
            r.pop("_symbols", None)
            r["n_unique_symbols"] = 0
        pd.DataFrame(rows, columns=["File"] + METRICS).to_csv(out_csv, index=False)
        messages_json_path.write_text("[]", encoding="utf-8")
        print(f"  [pylint_export] no files to analyze → empty CSV → {out_csv.name}")
