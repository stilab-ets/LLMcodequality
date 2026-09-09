"""
repo_manager.py — Git repository operations: clone, checkout, reset.
"""
import os
import subprocess
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

