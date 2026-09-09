"""
main.py — Entry point: run all models in parallel; within each model repos run in parallel;
          within each (model, repo) pair instances run sequentially.

Run from the project directory:
    python main.py                                          # all models, all repos
    python main.py --model claude-45haiku-10222025          # one model, all repos
    python main.py --repos qutebrowser,openlibrary          # all models, two repos
    python main.py --model claude-45sonnet-10132025 --repos ansible
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config             import UDB_DIR, REPO_ROOT, SUCCEEDED_DIR, REPO_MAP
from dataset_loader     import DatasetLoader
from pipeline_runner    import PipelineRunner
from metrics_aggregator import MetricsAggregator


def run_model(model: str, eval_dir: Path, target_repos: list[str] | None = None) -> None:
    out_dir = Path(f"understand_results/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"MODEL: {model}")
    print(f"  eval_dir : {eval_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  repo_root: {REPO_ROOT / model}")
    print(f"  udb_dir  : {UDB_DIR / model}")
    if target_repos:
        print(f"  repos    : {target_repos}")
    print(f"{'#' * 70}")

    valid_instances = DatasetLoader(eval_dir=eval_dir, target_repos=target_repos).load()
    if not valid_instances:
        print(f"  → no valid instances for {model}, skipping\n")
        return

    PipelineRunner(valid_instances, model=model, out_dir=out_dir).run()
    MetricsAggregator().aggregate_all(valid_instances, out_dir=out_dir)


def _resolve_repos(repos_arg: str | None) -> list[str] | None:
    if not repos_arg:
        return None
    name_to_id = {v: k for k, v in REPO_MAP.items()}
    result = []
    for token in repos_arg.split(","):
        token = token.strip()
        if token in REPO_MAP:
            result.append(token)
        elif token in name_to_id:
            result.append(name_to_id[token])
        else:
            raise SystemExit(
                f"Unknown repo '{token}'. Valid options: "
                + ", ".join(f"{v} ({k})" for k, v in REPO_MAP.items())
            )
    return result or None


def main():
    parser = argparse.ArgumentParser(description="SWE-Bench Pro Understand pipeline")
    parser.add_argument("--model", default=None,
                        help="Run only this model. Omit to run all models in parallel.")
    parser.add_argument("--repos", default=None,
                        help="Comma-separated repo short-names (e.g. 'ansible,openlibrary').")
    args = parser.parse_args()

    target_repos = _resolve_repos(args.repos)

    if args.model:
        eval_dir = SUCCEEDED_DIR / args.model
        if not eval_dir.is_dir():
            raise SystemExit(f"Model directory not found: {eval_dir}")
        run_model(model=args.model, eval_dir=eval_dir, target_repos=target_repos)
    else:
        model_dirs = sorted(d for d in SUCCEEDED_DIR.iterdir() if d.is_dir())
        print(f"Running {len(model_dirs)} model(s) in parallel: {[d.name for d in model_dirs]}")

        with ThreadPoolExecutor(max_workers=len(model_dirs)) as ex:
            futures = {
                ex.submit(run_model, d.name, d, target_repos): d.name
                for d in model_dirs
            }
            for f in as_completed(futures):
                model_name = futures[f]
                exc = f.exception()
                if exc:
                    print(f"[ERROR] {model_name}: {exc}")
                else:
                    print(f"[DONE]  {model_name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
