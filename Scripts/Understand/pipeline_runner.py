import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config import (
    OUT_DIR, UDB_DIR, REPO_ROOT, LANG_EXT,
    FILTER_PATCHES, TEST_MODE, REPO_MAP, REPO_LANG,
)
from repo_manager     import RepoManager
from udb_manager      import UdbManager
from patch_cleaner    import PatchCleaner
from patch_applier    import PatchApplier
from metrics_exporter import MetricsExporter
from diff_computer    import DiffComputer
from und_runner       import UndRunner


class InstanceProcessor:
    """Runs the full pipeline for a single instance. One per repo-thread."""

    def __init__(self, out_dir: Path, udb_dir: Path, repo_root: Path):
        runner        = UndRunner()
        self.out_dir  = out_dir
        self.udb_dir  = udb_dir
        self.repo_mgr = RepoManager(repo_root)
        self.udb_mgr  = UdbManager(udb_dir, runner)
        self.cleaner  = PatchCleaner()
        self.applier  = PatchApplier()
        self.exporter = MetricsExporter(runner)
        self.differ   = DiffComputer()

    def process(self, inst: dict, repo_path: Path) -> dict:
        iid       = inst["iid"]
        repo_name = inst["repo_name"]
        lang      = inst["lang"]
        exts      = LANG_EXT[lang]
        out_dir   = self.out_dir / repo_name / iid
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Instance: {iid}")

        has_llm_patch = inst["llm_patch_path"].exists()
        gold_done     = (out_dir / "diff_gold.csv").exists()
        llm_done      = (out_dir / "diff_llm.csv").exists() or not has_llm_patch
        if gold_done and llm_done:
            print("  already done — skipping")
            return {"iid": iid, "status": "skipped", "error": None}

        try:
            self.repo_mgr.checkout(repo_path, inst["base_commit"])

            gold_patch_text = inst["gold_patch_text"]
            llm_text        = ""
            gold_files, llm_files = [], []

            if gold_patch_text.strip():
                gold_files = self.applier.get_changed_files(repo_path, gold_patch_text, exts)
            if inst["llm_patch_path"].exists():
                llm_text  = inst["llm_patch_path"].read_text(encoding="utf-8", errors="ignore")
                llm_files = self.applier.get_changed_files(repo_path, llm_text, exts)

            patch_dir = inst["llm_patch_path"].parent
            if FILTER_PATCHES:
                stats_csv = out_dir / "patch_filter_stats.csv"
                gold_patch_text = self.cleaner.maybe_filter(
                    gold_patch_text, label="gold",
                    save_path=patch_dir / "filtered_gold.diff", stats_csv=stats_csv,
                )
                if llm_text:
                    llm_text = self.cleaner.maybe_filter(
                        llm_text, label="llm",
                        save_path=patch_dir / "filtered_llm.diff", stats_csv=stats_csv,
                    )

            safe_iid          = iid.replace("/", "_")
            gold_udb, llm_udb = self.udb_mgr.instance_udb_paths(safe_iid)
            miss_csv = out_dir / "udb_miss_stats.csv"

            print("\n── BUILD INSTANCE UDBs ──")
            if gold_files:
                self.udb_mgr.create_for_files(gold_udb, repo_path, gold_files, lang)
            if llm_files:
                self.udb_mgr.create_for_files(llm_udb, repo_path, llm_files, lang)

            print("\n── BEFORE METRICS ──")
            if gold_files:
                self.exporter.export(
                    gold_udb, gold_files, out_dir / "before_gold.csv",
                    miss_csv=miss_csv, label="before_gold",
                )
            if llm_files:
                self.exporter.export(
                    llm_udb, llm_files, out_dir / "before_llm.csv",
                    miss_csv=miss_csv, label="before_llm",
                )

            print("\n── APPLY + AFTER ──")
            if not (out_dir / "diff_gold.csv").exists():
                self._process_patch(
                    repo_path, gold_udb, gold_patch_text, gold_files, out_dir, "gold", miss_csv=miss_csv,
                )
            if not (out_dir / "diff_llm.csv").exists() and llm_text:
                self._process_patch(
                    repo_path, llm_udb, llm_text, llm_files, out_dir, "llm", miss_csv=miss_csv,
                )

            print("\n── DIFFS ──")
            for kind in ("gold", "llm"):
                b = out_dir / f"before_{kind}.csv"
                a = out_dir / f"after_{kind}.csv"
                d = out_dir / f"diff_{kind}.csv"
                if b.exists() and a.exists():
                    self.differ.compute(b, a, d)
                    b.unlink(missing_ok=True)
                    a.unlink(missing_ok=True)

            self.udb_mgr.cleanup_instance(gold_udb, llm_udb)

            print(f"\n[DONE] {iid}")
            return {"iid": iid, "status": "done", "error": None}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {iid}\n{tb}")
            return {"iid": iid, "status": "error", "error": str(e)}

    def _process_patch(self, repo_path, udb_path, patch_text, changed_files, out_dir, kind, miss_csv=None):
        if not changed_files:
            print(f"  [{kind}] no changed files — skipping")
            return

        print(f"\n── [{kind.upper()}] APPLY ──")
        tmp_path = self.applier.apply(repo_path, patch_text)
        if tmp_path is None:
            print(f"  [{kind}] patch failed — skipping AFTER metrics")
            return

        try:
            print(f"\n── [{kind.upper()}] AFTER ──")
            self.udb_mgr.runner.run(["analyze", "-changed", str(udb_path)], timeout=600)
            self.exporter.export(
                udb_path, changed_files, out_dir / f"after_{kind}.csv",
                miss_csv=miss_csv, label=f"after_{kind}",
            )
        finally:
            self.repo_mgr.reset(repo_path)
            tmp_path.unlink(missing_ok=True)


class PipelineRunner:
    """
    Parallelism layout:
      - Repos within a model run in parallel threads (one clone + sequential instances each).
      - Models themselves are parallelised from main.py.
    """

    def __init__(
        self,
        valid_instances: list[dict],
        model:     str,
        out_dir:   Path = OUT_DIR,
        udb_dir:   Path = UDB_DIR,
        test_mode: bool = TEST_MODE,
    ):
        self.valid_instances = valid_instances
        self.model     = model
        self.out_dir   = out_dir
        # Per-model isolated directories
        self.udb_dir   = udb_dir   / model
        self.repo_root = REPO_ROOT / model
        self.test_mode = test_mode

        self.udb_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict]:
        instances = self._first_per_repo() if self.test_mode else self.valid_instances

        # Group by repo
        by_repo: dict[str, list[dict]] = {}
        for inst in instances:
            by_repo.setdefault(inst["repo_name"], []).append(inst)

        print(
            f"\n[{self.model}] {len(instances)} instances "
            f"across {len(by_repo)} repo(s) — repos run in parallel"
        )

        # Clone all repos up-front (sequential, avoids concurrent git clones to same dir)
        repo_mgr   = RepoManager(self.repo_root)
        repo_paths: dict[str, Path] = {}
        for repo_id, repo_name in REPO_MAP.items():
            if repo_name not in by_repo:
                continue
            repo_paths[repo_name] = repo_mgr.clone_if_needed(repo_id, repo_name)

        all_results: list[dict] = []
        lock = threading.Lock()

        def run_repo(repo_name: str, repo_instances: list[dict]) -> None:
            repo_path = repo_paths[repo_name]
            processor = InstanceProcessor(self.out_dir, self.udb_dir, self.repo_root)
            for inst in repo_instances:
                res = processor.process(inst, repo_path)
                status_line = f"[{res['status'].upper():8s}] {res['iid']}"
                if res["error"]:
                    status_line += f"  — {res['error']}"
                print(f"[{self.model}/{repo_name}] {status_line}")
                with lock:
                    all_results.append(res)

        with ThreadPoolExecutor(max_workers=len(by_repo)) as ex:
            futures = {ex.submit(run_repo, rn, ri): rn for rn, ri in by_repo.items()}
            for f in as_completed(futures):
                repo_name = futures[f]
                exc = f.exception()
                if exc:
                    print(f"[{self.model}/{repo_name}] FATAL: {exc}")

        done    = sum(1 for r in all_results if r["status"] == "done")
        skipped = sum(1 for r in all_results if r["status"] == "skipped")
        errors  = sum(1 for r in all_results if r["status"] == "error")
        print(f"\n[{self.model}] Done={done}  Skipped={skipped}  Errors={errors}")
        return all_results

    def _first_per_repo(self) -> list[dict]:
        seen, out = set(), []
        for inst in self.valid_instances:
            if inst["repo_name"] not in seen:
                seen.add(inst["repo_name"])
                out.append(inst)
        return out
