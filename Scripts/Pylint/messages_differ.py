"""
messages_differ.py — Compute new/removed pylint messages between before and after a patch.

Matching strategy: multiset difference on (symbol, message_text) per file.
Line numbers are NOT used for identity because they shift after a patch is applied.
The actual line number of the matched record is kept in the output for reference.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Output columns for the messages diff CSV
MSG_COLUMNS = [
    "file",
    "status",       # "new" | "removed"
    "type",         # convention | refactor | warning | error | fatal
    "message-id",   # e.g. C0103
    "symbol",       # e.g. invalid-name
    "line",
    "column",
    "obj",          # class or function name, empty if module-level
    "message",      # full message text
]


class MessagesDiffer:
    """
    Computes the set of messages that were introduced (new) or resolved (removed)
    by a patch, by comparing pylint output before and after applying the patch.
    """

    def compute(
        self,
        before_json: Path,
        after_json:  Path,
        diff_csv:    Path,
    ) -> None:
        """
        Load the before/after message JSON files and write a CSV listing every
        new or removed message.

        - `removed`: appeared BEFORE, gone AFTER  → the patch fixed this issue
        - `new`:     appeared AFTER, not BEFORE    → the patch introduced this issue
        """
        print(f"  [msg_diff] {before_json.name} vs {after_json.name}")

        before_msgs = self._load(before_json)
        after_msgs  = self._load(after_json)

        # Group by file so we do per-file multiset diffing
        all_files = sorted(set(m["file"] for m in before_msgs) |
                           set(m["file"] for m in after_msgs))

        before_by_file = self._group_by_file(before_msgs)
        after_by_file  = self._group_by_file(after_msgs)

        diff_rows: list[dict] = []

        for filepath in all_files:
            b_msgs = before_by_file.get(filepath, [])
            a_msgs = after_by_file.get(filepath,  [])

            removed_records, new_records = self._multiset_diff(b_msgs, a_msgs)

            for rec in removed_records:
                diff_rows.append({**rec, "status": "removed", "file": filepath})
            for rec in new_records:
                diff_rows.append({**rec, "status": "new",     "file": filepath})

        diff_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(diff_rows, columns=MSG_COLUMNS) if diff_rows \
             else pd.DataFrame(columns=MSG_COLUMNS)
        df.to_csv(diff_csv, index=False)

        n_removed = sum(1 for r in diff_rows if r["status"] == "removed")
        n_new     = sum(1 for r in diff_rows if r["status"] == "new")
        print(f"  [msg_diff] removed={n_removed}  new={n_new} → {diff_csv.name}")

        if diff_rows:
            self._print_type_summary(diff_rows)

    # ── core diff logic ───────────────────────────────────────────────────────

    @staticmethod
    def _multiset_diff(
        before: list[dict],
        after:  list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Return (removed_records, new_records) using multiset arithmetic on
        the key (symbol, message_text).

        For each key with n_before occurrences and n_after occurrences:
          - If n_before > n_after: the first (n_before - n_after) records
            from `before` are tagged "removed".
          - If n_after > n_before: the first (n_after - n_before) records
            from `after` are tagged "new".
        """
        def key(m: dict) -> tuple[str, str]:
            return (m.get("symbol", ""), m.get("message", ""))

        before_counter = Counter(key(m) for m in before)
        after_counter  = Counter(key(m) for m in after)

        # Build lookup: key → deque of matching records (preserves order)
        from collections import deque
        before_by_key: dict[tuple, deque] = {}
        for m in before:
            k = key(m)
            before_by_key.setdefault(k, deque()).append(m)

        after_by_key: dict[tuple, deque] = {}
        for m in after:
            k = key(m)
            after_by_key.setdefault(k, deque()).append(m)

        removed_records: list[dict] = []
        new_records:     list[dict] = []

        all_keys = set(before_counter) | set(after_counter)
        for k in all_keys:
            n_b = before_counter[k]
            n_a = after_counter[k]
            if n_b > n_a:
                # More before than after: (n_b - n_a) were removed
                pool = list(before_by_key.get(k, []))
                removed_records.extend(pool[: n_b - n_a])
            elif n_a > n_b:
                # More after than before: (n_a - n_b) are new
                pool = list(after_by_key.get(k, []))
                new_records.extend(pool[: n_a - n_b])

        # Sort for deterministic output: by type severity then line
        _type_order = {"fatal": 0, "error": 1, "warning": 2, "refactor": 3, "convention": 4}
        for lst in (removed_records, new_records):
            lst.sort(key=lambda m: (
                _type_order.get(m.get("type", ""), 9),
                m.get("file", ""),
                m.get("line") or 0,
            ))

        return removed_records, new_records

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _group_by_file(messages: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for m in messages:
            groups.setdefault(m.get("file", ""), []).append(m)
        return groups

    @staticmethod
    def _print_type_summary(rows: list[dict]) -> None:
        from collections import defaultdict
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            counts[r["status"]][r.get("type", "?")] += 1
        for status in ("removed", "new"):
            if counts[status]:
                parts = "  ".join(f"{t}={n}" for t, n in sorted(counts[status].items()))
                print(f"    {status}: {parts}")
