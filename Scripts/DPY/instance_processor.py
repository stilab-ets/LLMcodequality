"""
instance_processor.py — Full DPy pipeline for a single instance.
"""
import shutil
import traceback
from pathlib import Path

from config import OUT_DIR, LANG_EXT, FILTER_PATCHES
from dpy_runner    import DpyRunner
from repo_manager  import RepoManager
from patch_cleaner import PatchCleaner
from dpy_analyzer  import DpyAnalyzer
from diff_computer import DiffComputer


class InstanceProcessor:
    """
    Runs the full DPy pipeline for a single SWE-Bench instance:
      For each patch (gold, llm):
        1. Detect changed files → compute common ancestor folder
        2. Run DPy BEFORE on that folder
        3. Apply patch → Run DPy AFTER on the same folder → diff → save
        4. Reset repo to base commit
    """

    def __init__(self, out_dir: Path = OUT_DIR):
        self.out_dir   = out_dir
        self.runner    = DpyRunner()
        self.repo_mgr  = RepoManager()
        self.cleaner   = PatchCleaner()
        self.analyzer  = DpyAnalyzer()
        self.differ    = DiffComputer()

    def process(self, inst: dict) -> dict:
        iid       = inst["iid"]
        repo_name = inst["repo_name"]
        exts      = LANG_EXT[inst["lang"]]
        out_dir   = self.out_dir / repo_name / iid
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Instance: {iid}")

        # Skip if already done
        if all((out_dir / f"diff_gold_{Path(f).stem}.csv").exists() for f in
               ["function_metrics.csv", "class_module_metrics.csv"]):
            print("  already done — skipping")
            return {"iid": iid, "status": "skipped", "error": None}

        repo_path = self.repo_mgr.clone_if_needed(inst["repo_id"], repo_name)

        try:
            # ── Checkout ──────────────────────────────────────────────────
            self.repo_mgr.checkout(repo_path, inst["base_commit"])

            # ── Detect changed files ──────────────────────────────────────
            gold_patch_text = inst["gold_patch_text"]
            llm_text        = ""
            gold_files, llm_files = [], []

            if gold_patch_text.strip():
                gold_files = self.repo_mgr.get_changed_files(
                    repo_path, gold_patch_text, exts
                )
            if inst["llm_patch_path"].exists():
                llm_text  = inst["llm_patch_path"].read_text(encoding="utf-8", errors="ignore")
                llm_files = self.repo_mgr.get_changed_files(repo_path, llm_text, exts)

            # ── Filter patches ────────────────────────────────────────────
            patch_dir = inst["llm_patch_path"].parent
            stats_csv = out_dir / "patch_filter_stats.csv"
            if gold_patch_text.strip():
                gold_patch_text = self.cleaner.maybe_filter(
                    gold_patch_text, label="gold",
                    save_path=patch_dir / "filtered_gold.diff",
                    stats_csv=stats_csv,
                )
            if llm_text:
                llm_text = self.cleaner.maybe_filter(
                    llm_text, label="llm",
                    save_path=patch_dir / "filtered_llm.diff",
                    stats_csv=stats_csv,
                )

            # ── GOLD: BEFORE → apply → AFTER → reset ─────────────────────
            if gold_files and gold_patch_text.strip():
                self._run_patch(repo_path, gold_patch_text, gold_files, out_dir, "gold")

            # ── LLM:  BEFORE → apply → AFTER → reset ─────────────────────
            if llm_files and llm_text.strip():
                self._run_patch(repo_path, llm_text, llm_files, out_dir, "llm")

            # ── Cleanup ───────────────────────────────────────────────────
            shutil.rmtree(self.out_dir / "_tmp" / iid, ignore_errors=True)

            print(f"\n[DONE] {iid}")
            return {"iid": iid, "status": "done", "error": None}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {iid}\n{tb}")
            return {"iid": iid, "status": "error", "error": str(e)}

    def _analyze_dirs(self, dirs: list, iid: str, phase: str) -> dict:
        """Run DPy on each directory and return concatenated results."""
        out_dirs = []
        for i, adir in enumerate(dirs):
            print(f"  [analyze] {phase} [{i+1}/{len(dirs)}] → {adir}")
            out = self.out_dir / "_tmp" / iid / f"{phase}_{i}"
            if not self.runner.analyze(adir, out):
                raise RuntimeError(f"DPy {phase} analysis failed on {adir}")
            out_dirs.append(out)
        return self.analyzer.load_and_concat_csvs(out_dirs)

    def _run_patch(
        self,
        repo_path:     Path,
        patch_text:    str,
        changed_files: list[str],
        out_dir:       Path,
        kind:          str,
    ) -> None:
        """BEFORE → apply patch → AFTER (same dirs) → diff → reset."""
        iid           = out_dir.name
        analysis_dirs = self.repo_mgr.get_analysis_dirs(repo_path, changed_files)

        # ── BEFORE ───────────────────────────────────────────────────────
        print(f"\n── [{kind.upper()}] BEFORE ──")
        before_dfs_full = self._analyze_dirs(analysis_dirs, iid, f"before_{kind}")

        # ── APPLY ────────────────────────────────────────────────────────
        print(f"\n── [{kind.upper()}] APPLY ──")
        tmp_path = self.repo_mgr.apply_patch(repo_path, patch_text)
        if tmp_path is None:
            print(f"  [{kind}] patch failed — skipping AFTER")
            return

        try:
            # ── AFTER (same dirs as BEFORE) ───────────────────────────────
            print(f"\n── [{kind.upper()}] AFTER ──")
            try:
                after_dfs_full = self._analyze_dirs(analysis_dirs, iid, f"after_{kind}")
            except RuntimeError:
                print(f"  [{kind}] DPy AFTER failed — skipping diff")
                return

            before_filtered = self.analyzer.filter_to_changed(before_dfs_full, changed_files)
            after_filtered  = self.analyzer.filter_to_changed(after_dfs_full,  changed_files)

            # ── DIFF: introduced = in after not before; removed = in before not after ──
            print(f"\n── [{kind.upper()}] DIFF ──")
            diffs = self.differ.compute(before_filtered, after_filtered)
            self.differ.save(diffs, before_filtered, after_filtered, out_dir, kind)

        finally:
            self.repo_mgr.reset(repo_path)
            tmp_path.unlink(missing_ok=True)
