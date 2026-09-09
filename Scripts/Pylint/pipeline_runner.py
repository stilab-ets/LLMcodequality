"""
pipeline_runner.py — Full pipeline: per-instance processor, per-repo worker, parallel runner.
"""
import os
import sys
import shutil
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config import (
    OUT_DIR, LANG_EXT, FILTER_PATCHES, TEST_MODE, N_WORKERS,
)
from repo_manager    import RepoManager
from patch_cleaner   import PatchCleaner
from patch_applier   import PatchApplier
from pylint_runner   import PylintRunner
from pylint_exporter import PylintExporter
from diff_computer   import DiffComputer
from messages_differ import MessagesDiffer


# ── Logging helper ────────────────────────────────────────────────────────────

class _Tee:
    """Redirect all print() calls inside a worker to a log file while preserving stdout."""

    def __init__(self, path: Path):
        self._path        = path
        self._file        = None
        self._orig_stdout = None
        self._orig_stderr = None

    def write(self, msg):
        self._file.write(msg)
        self._file.flush()
        self._orig_stdout.write(msg)

    def flush(self):
        self._file.flush()
        self._orig_stdout.flush()

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file        = open(self._path, "w", encoding="utf-8", buffering=1)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._file.close()


# ── Per-instance logic ────────────────────────────────────────────────────────

class InstanceProcessor:
    """Runs the full pylint pipeline for a single instance."""

    def __init__(self, out_dir: Path = OUT_DIR):
        self.out_dir  = out_dir
        self.repo_mgr   = RepoManager()
        self.cleaner    = PatchCleaner()
        self.applier    = PatchApplier()
        runner          = PylintRunner()
        self.exporter   = PylintExporter(runner)
        self.differ     = DiffComputer()
        self.msg_differ = MessagesDiffer()

    def process(self, inst: dict, repo_path: Path) -> dict:
        """Run the full pylint pipeline for one instance."""
        iid       = inst["iid"]
        repo_name = inst["repo_name"]
        lang      = inst["lang"]
        exts      = LANG_EXT[lang]
        out_dir   = self.out_dir / repo_name / iid
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Instance: {iid}")

        already_done = (
            (out_dir / "diff_gold.csv").exists() and
            (out_dir / "diff_llm.csv").exists() and
            (out_dir / "messages_diff_gold.csv").exists() and
            (out_dir / "messages_diff_llm.csv").exists()
        )
        if already_done:
            print("  already done — skipping")
            return {"iid": iid, "status": "skipped", "error": None}

        try:
            # ── Checkout base commit ──────────────────────────────────────
            self.repo_mgr.checkout(repo_path, inst["base_commit"])

            # ── Detect changed files ──────────────────────────────────────
            gold_patch_text = inst["gold_patch_text"]
            llm_text        = ""
            gold_files, llm_files = [], []

            if gold_patch_text.strip():
                gold_files = self.applier.get_changed_files(
                    repo_path, gold_patch_text, exts
                )
            if inst["llm_patch_path"].exists():
                llm_text  = inst["llm_patch_path"].read_text(encoding="utf-8", errors="ignore")
                llm_files = self.applier.get_changed_files(repo_path, llm_text, exts)

            # ── Filter patches ────────────────────────────────────────────
            patch_dir = inst["llm_patch_path"].parent
            if FILTER_PATCHES:
                stats_csv = out_dir / "patch_filter_stats.csv"
                gold_patch_text = self.cleaner.maybe_filter(
                    gold_patch_text,
                    label="gold",
                    save_path=patch_dir / "filtered_gold.diff",
                    stats_csv=stats_csv,
                )
                if llm_text:
                    llm_text = self.cleaner.maybe_filter(
                        llm_text,
                        label="llm",
                        save_path=patch_dir / "filtered_llm.diff",
                        stats_csv=stats_csv,
                    )

            # ── BEFORE metrics (base commit state) ────────────────────────
            print("\n── BEFORE METRICS ──")
            if gold_files:
                self.exporter.export(repo_path, gold_files, out_dir / "before_gold.csv")
            if llm_files:
                self.exporter.export(repo_path, llm_files,  out_dir / "before_llm.csv")

            # ── Apply gold patch + AFTER metrics ──────────────────────────
            print("\n── APPLY + AFTER ──")
            if not (out_dir / "diff_gold.csv").exists():
                self._process_patch(
                    repo_path, gold_patch_text, gold_files, out_dir, "gold"
                )

            # ── Apply LLM patch + AFTER metrics ───────────────────────────
            if not (out_dir / "diff_llm.csv").exists() and llm_text:
                self._process_patch(
                    repo_path, llm_text, llm_files, out_dir, "llm"
                )

            # ── Compute count diffs + message diffs ───────────────────────
            print("\n── DIFFS ──")
            for kind in ("gold", "llm"):
                b_csv  = out_dir / f"before_{kind}.csv"
                a_csv  = out_dir / f"after_{kind}.csv"
                d_csv  = out_dir / f"diff_{kind}.csv"
                b_json = out_dir / f"before_{kind}.messages.json"
                a_json = out_dir / f"after_{kind}.messages.json"
                md_csv = out_dir / f"messages_diff_{kind}.csv"

                if b_csv.exists() and a_csv.exists():
                    self.differ.compute(b_csv, a_csv, d_csv)
                    b_csv.unlink(missing_ok=True)
                    a_csv.unlink(missing_ok=True)

                if b_json.exists() and a_json.exists():
                    self.msg_differ.compute(b_json, a_json, md_csv)
                    b_json.unlink(missing_ok=True)
                    a_json.unlink(missing_ok=True)

            print(f"\n[DONE] {iid}")
            return {"iid": iid, "status": "done", "error": None}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {iid}\n{tb}")
            return {"iid": iid, "status": "error", "error": str(e)}

    def _process_patch(
        self,
        repo_path:     Path,
        patch_text:    str,
        changed_files: list[str],
        out_dir:       Path,
        kind:          str,
    ) -> None:
        """Apply patch → run pylint → export AFTER metrics."""
        if not changed_files:
            print(f"  [{kind}] no changed files — skipping")
            return

        print(f"\n── [{kind.upper()}] APPLY ──")
        tmp_path = self.applier.apply(repo_path, patch_text)
        if tmp_path is None:
            print(f"  [{kind}] patch failed — skipping AFTER metrics")
            return

        try:
            print(f"\n── [{kind.upper()}] AFTER METRICS ──")
            self.exporter.export(repo_path, changed_files, out_dir / f"after_{kind}.csv")
        finally:
            self.repo_mgr.reset(repo_path)
            tmp_path.unlink(missing_ok=True)


# ── Per-repo worker (runs in a subprocess) ────────────────────────────────────

def _process_repo_worker(
    repo_name:      str,
    repo_instances: list,
    out_dir:        Path,
    log_dir:        Path,
) -> list[dict]:
    """
    Worker function for one repo. Runs in a subprocess.
    Processes all instances of the repo SEQUENTIALLY.
    """
    repo_log  = log_dir / f"{repo_name}_worker.log"
    results   = []

    with _Tee(repo_log):
        print(f"{'#' * 60}")
        print(f"REPO WORKER: {repo_name}  ({len(repo_instances)} instances)  PID={os.getpid()}")
        print(f"{'#' * 60}")

        repo_mgr  = RepoManager()
        processor = InstanceProcessor(out_dir)

        try:
            repo_id   = repo_instances[0]["repo_id"]
            base_repo = repo_mgr.clone_if_needed(repo_id, repo_name)
            repo_path = repo_mgr.make_worker_clone(base_repo, repo_name)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[FATAL] Could not set up worker repo for {repo_name}:\n{tb}")
            return [
                {"iid": inst["iid"], "status": "error", "error": str(e)}
                for inst in repo_instances
            ]

        for inst in repo_instances:
            res = processor.process(inst, repo_path)
            results.append(res)

        shutil.rmtree(str(repo_path), ignore_errors=True)
        print(f"\n[REPO DONE] {repo_name} — cleaned up worker repo")

    return results


# ── Top-level pipeline runner ─────────────────────────────────────────────────

class PipelineRunner:
    """
    Orchestrates the full Pylint pipeline.
    Spawns N_WORKERS subprocesses (one per repo) to process instances in parallel.
    No pre-build step is needed — pylint is run directly on source files.
    """

    def __init__(
        self,
        valid_instances: list[dict],
        out_dir:   Path = OUT_DIR,
        n_workers: int  = N_WORKERS,
        test_mode: bool = TEST_MODE,
    ):
        self.valid_instances = valid_instances
        self.out_dir   = out_dir
        self.n_workers = n_workers
        self.test_mode = test_mode
        self.log_dir   = out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.master_log = self.log_dir / "pipeline.log"

    def run(self) -> list[dict]:
        """Run the full pipeline and return a list of per-instance result dicts."""
        instances_to_run = (
            self._first_per_repo() if self.test_mode else self.valid_instances
        )

        repo_groups: dict[str, list] = defaultdict(list)
        for inst in instances_to_run:
            repo_groups[inst["repo_name"]].append(inst)

        print(f"\nPipeline: {len(instances_to_run)} instances across {len(repo_groups)} repos")
        print(f"  Workers: {self.n_workers} (one per repo, sequential within each repo)")
        for rname, rinsts in repo_groups.items():
            print(f"  {rname}: {len(rinsts)} instances")
        print(f"  Logs: {self.log_dir}")

        all_results: list[dict] = []

        with open(self.master_log, "w", encoding="utf-8", buffering=1) as master_log:
            master_log.write(
                f"Pipeline started — {len(instances_to_run)} instances, "
                f"{len(repo_groups)} repo-workers\n\n"
            )

            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {
                    pool.submit(
                        _process_repo_worker,
                        repo_name, repo_insts,
                        self.out_dir, self.log_dir,
                    ): repo_name
                    for repo_name, repo_insts in repo_groups.items()
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc="repos"):
                    repo_name = futures[future]
                    try:
                        repo_results = future.result()
                    except Exception as e:
                        tb = traceback.format_exc()
                        repo_results = [
                            {"iid": f"{repo_name}/?", "status": "error", "error": str(e)}
                        ]

                    for res in repo_results:
                        all_results.append(res)
                        status_line = f"[{res['status'].upper():8s}] {res['iid']}"
                        if res["error"]:
                            status_line += f"  — {res['error']}"
                        print(status_line)
                        master_log.write(status_line + "\n")

        done    = sum(1 for r in all_results if r["status"] == "done")
        skipped = sum(1 for r in all_results if r["status"] == "skipped")
        errors  = sum(1 for r in all_results if r["status"] == "error")
        print(f"\nDone={done}  Skipped={skipped}  Errors={errors}")
        print(f"Repo logs: {self.log_dir}/<repo>_worker.log")
        print(f"Master log: {self.master_log}")
        return all_results

    def _first_per_repo(self) -> list[dict]:
        seen, out = set(), []
        for inst in self.valid_instances:
            if inst["repo_name"] not in seen:
                seen.add(inst["repo_name"])
                out.append(inst)
        return out
