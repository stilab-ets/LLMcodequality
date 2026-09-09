"""
repo_manager.py — Git repository operations: clone, checkout, reset, worker clone.
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

    def make_worker_clone(self, base_repo: Path, repo_name: str) -> Path:
        """
        Create one isolated worker clone per repo using git clone --local.
        Uses hardlinks for speed and handles submodules correctly.
        """
        worker_repo = self.repo_root / f"{repo_name}_worker_{os.getpid()}"
        if worker_repo.exists():
            return worker_repo

        print(f"  [repo] git clone --local {repo_name} → {worker_repo.name}")
        r = subprocess.run(
            ["git", "clone", "--local", str(base_repo), str(worker_repo)],
            text=True, capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone --local failed:\n{r.stderr}")

        r2 = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=worker_repo, text=True, capture_output=True,
        )
        if r2.returncode != 0:
            print(f"  [repo] WARNING: submodule init failed:\n{r2.stderr[:200]}")

        print(f"  [repo] OK → {worker_repo}")
        return worker_repo
