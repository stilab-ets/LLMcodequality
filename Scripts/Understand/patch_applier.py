"""
patch_applier.py — Detect changed files from a patch and apply it to the repo.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path


class PatchApplier:
    """Applies git patches and detects which source files they touch."""

    def get_changed_files(
        self, repo_path: Path, patch_text: str, exts: set[str]
    ) -> list[str]:
        """
        Return the sorted, deduplicated list of source files touched by a patch,
        filtered by extension.

        Strategy 1: git apply --numstat  (preferred)
        Strategy 2: parse diff --git headers  (fallback)
        """
        clean = patch_text.replace("\r\n", "\n").replace("\r", "\n")
        fd, tmp = tempfile.mkstemp(suffix=".diff")
        os.close(fd)
        tmp_path = Path(tmp)
        tmp_path.write_text(clean, encoding="utf-8")

        files = []

        r = subprocess.run(
            f'git apply --numstat "{tmp_path}"',
            cwd=repo_path, shell=True, text=True, capture_output=True,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) == 3:
                    p = parts[2].replace("\\", "/").lstrip("ab/")
                    if Path(p).suffix in exts:
                        files.append(p)

        if not files:
            print("  [changed_files] numstat failed, falling back to diff header parse")
            for line in clean.splitlines():
                if line.startswith("diff --git "):
                    m = re.match(r"diff --git a/(.+) b/(.+)", line)
                    if m:
                        for p in (m.group(1), m.group(2)):
                            if Path(p).suffix in exts:
                                files.append(p)

        tmp_path.unlink(missing_ok=True)
        files = sorted(set(files))
        print(f"  [changed_files] {len(files)} file(s): {files[:5]}")
        return files

    def apply(self, repo_path: Path, patch_text: str) -> Path | None:
        """
        Apply a patch using three strategies in order.
        Returns the temp diff Path on success, or None on failure.
        """
        clean = patch_text.replace("\r\n", "\n").replace("\r", "\n")
        fd, tmp = tempfile.mkstemp(suffix=".diff")
        os.close(fd)
        tmp_path = Path(tmp)
        tmp_path.write_text(clean, encoding="utf-8")

        for cmd in (
            f'git apply "{tmp_path}"',
            f'git apply --ignore-space-change --ignore-whitespace "{tmp_path}"',
            f'git apply --3way "{tmp_path}"',
        ):
            r = subprocess.run(
                cmd, cwd=repo_path, shell=True, text=True, capture_output=True,
            )
            if r.returncode == 0:
                print("  [apply] OK")
                return tmp_path

        print("  [apply] FAILED — all strategies exhausted")
        tmp_path.unlink(missing_ok=True)
        return None