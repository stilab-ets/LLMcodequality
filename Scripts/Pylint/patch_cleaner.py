"""
patch_cleaner.py — Filter non-production hunks from unified diffs.
"""
import csv
import re
from pathlib import Path

from config import FILTER_PATCHES


class PatchCleaner:
    """
    Cleans git diffs by removing newly-added top-level files and files
    matching known non-production patterns (tests, docs, CI, etc.).
    """

    AUXILIARY_PATTERNS = [
        r'^IMPLEMENTATION_SUMMARY\.md$',
        r'^demo_.*\.py$',
        r'^example_.*\.py$',
        r'^implement_.*\.py$',
        r'^setup_.*\.py$',
        r'^[^/]+_demo\.py$',
        r'^[^/]+_example\.py$',
        r'^[^/]+_implementation\.py$',
        r'^test_implementation\.py$',
        r'^verify_.*\.py$',
        r'^validation_.*\.py$',
        r'^check_.*\.py$',
        r'^run_.*\.py$',
        r'^quick_test\.py$',
        r'^usage_.*\.py$',
        r'^[^/]+\.txt$',
        r'^notes\.md$',
        r'^TODO\.md$',
        r'^NOTES\.md$',
        r"(^|/)test_.*\.py$",
        r"(^|/).*_test\.py$",
        r"(^|/)tests?/",
        r"(^|/)tests\.py$",
        r"(^|/)test\.py$",
        r"(^|/)test_implementation\.py$",
        # Root-anchored: these names target agent-generated scratch scripts at the
        # repository root. A *nested* validation.py is production code — e.g.
        # lib/ansible/module_utils/common/validation.py — and must not be dropped.
        r"^final_validation\.py$",
        r"^validation\.py$",
        r"(^|/)docs?/",
        r"\.(md|rst|txt|adoc|asciidoc)$",
        r"\.(yml|yaml)$",
    ]

    def __init__(self):
        self._aux_regex = [re.compile(p, re.IGNORECASE) for p in self.AUXILIARY_PATTERNS]

    def is_auxiliary_file(self, filepath: str) -> bool:
        p = filepath.replace("\\", "/").lstrip("ab/")
        return any(pat.search(p) for pat in self._aux_regex)

    def is_top_level_file(self, filepath: str) -> bool:
        return "/" not in filepath.replace("\\", "/").lstrip("ab/")

    def _extract_path(self, header: str) -> str | None:
        m = re.match(r"^diff --git a/(.+) b/(.+)$", header)
        return m.group(2) if m else None

    def clean_diff(self, patch_text: str) -> tuple[str, list[str], list[str]]:
        """
        Split a unified diff into kept and dropped file blocks.

        A file block is dropped when EITHER condition holds:
          • it is a *newly added* file at the repo root, OR
          • its path matches any entry in AUXILIARY_PATTERNS.
        """
        lines = patch_text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)

        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if line.startswith("diff --git ") and current:
                blocks.append(current)
                current = []
            current.append(line)
        if current:
            blocks.append(current)

        kept_files:    list[str] = []
        dropped_files: list[str] = []
        kept_blocks:   list[list[str]] = []

        for block in blocks:
            header    = block[0].rstrip("\n")
            file_path = self._extract_path(header)

            if file_path is None:
                kept_blocks.append(block)
                continue

            is_new_file = any("new file mode" in line for line in block[:6])

            if (is_new_file and self.is_top_level_file(file_path)) or \
               self.is_auxiliary_file(file_path):
                dropped_files.append(file_path)
            else:
                kept_files.append(file_path)
                kept_blocks.append(block)

        filtered_patch = "".join("".join(b) for b in kept_blocks)
        return filtered_patch, kept_files, dropped_files

    def maybe_filter(
        self,
        patch_text: str,
        label: str = "",
        save_path: Path | None = None,
        stats_csv: Path | None = None,
    ) -> str:
        if not FILTER_PATCHES:
            return patch_text

        filtered, kept, dropped = self.clean_diff(patch_text)

        tag = f"[{label}] " if label else ""
        print(f"  {tag}patch filter: kept {len(kept)} file(s), dropped {len(dropped)} file(s)")
        for f in dropped:
            print(f"    ✗ dropped: {f}")
        for f in kept:
            print(f"    ✓ kept:    {f}")

        if not kept:
            print(f"  {tag}WARNING: all hunks were filtered out — patch is empty")

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(filtered, encoding="utf-8")
            print(f"  {tag}filtered patch saved → {save_path}")

        if stats_csv is not None:
            stats_csv.parent.mkdir(parents=True, exist_ok=True)
            rows = (
                [{"file": f, "status": "kept",    "patch": label} for f in kept] +
                [{"file": f, "status": "dropped", "patch": label} for f in dropped]
            )
            write_header = not stats_csv.exists()
            with open(stats_csv, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["file", "status", "patch"])
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
            print(f"  {tag}filter stats saved → {stats_csv}")

        return filtered
