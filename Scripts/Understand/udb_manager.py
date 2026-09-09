"""
udb_manager.py — SciTools Understand database creation, settings, and copying.
"""
import shutil
from pathlib import Path

from config import UDB_DIR, UDB_SETTINGS
from und_runner import UndRunner


class UdbManager:
    """Creates, configures, and copies Understand (.und) databases."""

    def __init__(self, udb_dir: Path = UDB_DIR, runner: UndRunner = None):
        self.udb_dir = udb_dir
        self.udb_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or UndRunner()

    def create_base(self, udb_path: Path, repo_path: Path, lang: str) -> None:
        """
        Create a fresh .und database, configure it, add the full repo,
        and run a complete analysis. Built once per repo.
        """
        print(f"  [und] creating base UDB: {udb_path.name}  (lang={lang})")
        shutil.rmtree(str(udb_path), ignore_errors=True)
        udb_path.with_suffix(".csv").unlink(missing_ok=True)

        self.runner.run(["create", "-languages", lang, str(udb_path)])

        settings_args = ["settings"]
        for k, v in UDB_SETTINGS.items():
            settings_args += [k, v]
        settings_args.append(str(udb_path))
        self.runner.run(settings_args)

        self.runner.run(["add", str(repo_path), str(udb_path)])
        print("  [analyze] full analysis (-all)...")
        self.runner.run(["analyze", "-all", str(udb_path)], timeout=600)
        print(f"  [OK] base UDB ready: {udb_path}")

    def create_for_files(
        self,
        udb_path:  Path,
        repo_path: Path,
        files:     list[str],
        lang:      str,
    ) -> None:
        """Create a small UDB containing only the specified files and analyze them."""
        shutil.rmtree(str(udb_path), ignore_errors=True)
        udb_path.with_suffix(".csv").unlink(missing_ok=True)

        self.runner.run(["create", "-languages", lang, str(udb_path)])

        settings_args = ["settings"]
        for k, v in UDB_SETTINGS.items():
            settings_args += [k, v]
        settings_args.append(str(udb_path))
        self.runner.run(settings_args)

        for f in files:
            full = repo_path / f
            if full.exists():
                self.runner.run(["add", str(full), str(udb_path)])

        self.runner.run(["analyze", "-all", str(udb_path)], timeout=600)

    def copy(self, src_udb: Path, dst_udb: Path) -> None:
        """
        Copy a .und database directory to a new path.
        Understand databases are directories, so shutil.copytree is used.
        Any existing destination is removed first.
        """
        shutil.rmtree(str(dst_udb), ignore_errors=True)
        dst_udb.with_suffix(".csv").unlink(missing_ok=True)
        shutil.copytree(str(src_udb), str(dst_udb))
        print(f"  [copy_udb] {src_udb.name} → {dst_udb.name}")

    def instance_udb_paths(self, safe_iid: str) -> tuple[Path, Path]:
        """Return (gold_udb, llm_udb) paths for a given instance id."""
        gold_udb = self.udb_dir / f"{safe_iid}_gold.und"
        llm_udb  = self.udb_dir / f"{safe_iid}_llm.und"
        return gold_udb, llm_udb

    def cleanup_instance(self, gold_udb: Path, llm_udb: Path) -> None:
        """Remove per-instance UDB copies and their CSV sidecars."""
        for udb in (gold_udb, llm_udb):
            shutil.rmtree(str(udb), ignore_errors=True)
            udb.with_suffix(".csv").unlink(missing_ok=True)