"""
repo_manager.py — Git repository operations: clone, checkout, reset, apply patch.
"""
import os
import subprocess
import tempfile
from pathlib import Path

from config import REPO_ROOT


class RepoManager:
    """Manages local git repository clones for the pipeline."""

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.repo_root.mkdir(parents=True, exist_ok=True)

    def clone_if_needed(self, repo_id: str, repo_name: str) -> Path:
        """Clone the GitHub repository if not already present locally."""
        repo_path = self.repo_root / repo_name
        if repo_path.is_dir():
            print(f"  [clone] already exists: {repo_path}")
            return repo_path
        owner, name = repo_id.split("__")
        url = f"https://github.com/{owner}/{name}.git"
        print(f"  [clone] cloning {url}")
        r = subprocess.run(
            ["git", "clone", url, str(repo_path)],
            text=True, capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{r.stderr}")
        print("  [clone] OK")
        return repo_path

    def checkout(self, repo_path: Path, commit: str) -> None:
        """Hard-reset the repo to a specific base commit."""
        print(f"  [git] checkout {commit[:10]}...")
        for cmd in (
            ["git", "reset", "--hard"],
            ["git", "clean", "-fd"],
            ["git", "checkout", commit],
        ):
            r = subprocess.run(cmd, cwd=repo_path, text=True, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
        print("  [git] OK")

    def reset(self, repo_path: Path) -> None:
        """Hard-reset the repo to discard any applied patch."""
        subprocess.run(["git", "reset", "--hard"], cwd=repo_path,
                       text=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],    cwd=repo_path,
                       text=True, capture_output=True)
        print("  [git] repo reset to HEAD")

    def apply_patch(self, repo_path: Path, patch_text: str) -> Path | None:
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
            r = subprocess.run(cmd, cwd=repo_path, shell=True,
                               text=True, capture_output=True)
            if r.returncode == 0:
                print("  [apply] OK")
                return tmp_path

        print("  [apply] FAILED — all strategies exhausted")
        tmp_path.unlink(missing_ok=True)
        return None

    def get_analysis_dirs(self, repo_path: Path, changed_files: list[str]) -> list[Path]:
        """
        Return the minimal set of directories to pass to DPy:
        - If all changed files share a common subdirectory below the repo root, return [that dir].
        - Otherwise return each unique top-level subdirectory that contains changed files.
        """
        if not changed_files:
            return [repo_path]

        abs_parents = sorted({
            (repo_path / Path(f).parent).resolve()
            for f in changed_files
        })
        abs_parents = [p for p in abs_parents if p.exists()]
        if not abs_parents:
            return [repo_path]

        try:
            common = Path(os.path.commonpath([str(p) for p in abs_parents])).resolve()
        except ValueError:
            common = repo_path.resolve()

        repo_resolved = repo_path.resolve()
        if common != repo_resolved and str(common).startswith(str(repo_resolved)):
            rel = common.relative_to(repo_resolved)
            print(f"  [analysis_dirs] common folder: {rel}")
            return [common]

        # No single common subfolder — collect unique top-level subdirs
        subdirs: set[Path] = set()
        for f in changed_files:
            parts = Path(f).parts
            subdirs.add(repo_path / parts[0] if len(parts) > 1 else repo_path)

        result = sorted({d for d in subdirs if d.exists()}, key=str)
        rels = [str(d.relative_to(repo_path)) for d in result]
        print(f"  [analysis_dirs] {len(result)} folder(s): {rels}")
        return result if result else [repo_path]

    def get_changed_files(
        self, repo_path: Path, patch_text: str, exts: set[str]
    ) -> list[str]:
        """
        Return sorted deduplicated list of source files touched by a patch,
        filtered by extension.
        Strategy 1: git apply --numstat (preferred)
        Strategy 2: parse diff --git headers (fallback)
        """
        import re
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