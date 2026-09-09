"""
main.py — Entry point for the Pylint static-analysis pipeline.

Run from the Scripts/Pylint/ directory:
    python main.py                                           # all models, all repos
    python main.py --model claude-45haiku-10222025           # one model, all repos
    python main.py --repos qutebrowser,openlibrary           # all models, two repos
    python main.py --model claude-45sonnet-10132025 --repos ansible
    python main.py --test                                    # one instance per repo (smoke test)
"""
import argparse
from pathlib import Path

from config             import SUCCEEDED_DIR, REPO_MAP
from dataset_loader     import DatasetLoader
from pipeline_runner    import PipelineRunner
from metrics_aggregator import MetricsAggregator


def run_model(model: str, eval_dir: Path, target_repos: list[str] | None = None) -> None:
    out_dir = Path(f"pylint_results/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"MODEL: {model}")
    print(f"  eval_dir    : {eval_dir}")
    print(f"  out_dir     : {out_dir}")
    if target_repos:
        print(f"  repos filter: {target_repos}")
    print(f"{'#' * 70}")

    # ── Step 1: Load dataset ─────────────────────────────────────────────
    valid_instances = DatasetLoader(eval_dir=eval_dir, target_repos=target_repos).load()
    if not valid_instances:
        print(f"  → no valid instances for {model}, skipping\n")
        return

    # ── Step 2: Run pipeline (checkout → pylint before → apply → pylint after) ──
    runner = PipelineRunner(valid_instances, out_dir=out_dir)
    runner.run()

    # ── Step 3: Aggregate metrics ──────────────────────────────────────────
    MetricsAggregator().aggregate_all(valid_instances, out_dir=out_dir)


def _resolve_repos(repos_arg: str | None) -> list[str] | None:
    """
    Convert a comma-separated repo shortname string to a list of repo_ids.
    Accepts both short names (e.g. "ansible") and full ids (e.g. "ansible__ansible").
    Returns None if no filter was requested (= all repos).
    """
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
    parser = argparse.ArgumentParser(description="SWE-Bench Pro Pylint pipeline")
    parser.add_argument(
        "--model", default=None,
        help="Run only this model (directory name under python_data/succeeded/). "
             "Omit to run all models.",
    )
    parser.add_argument(
        "--repos", default=None,
        help="Comma-separated repo short-names or repo_ids to process "
             "(e.g. 'ansible,openlibrary'). Omit to process all repos.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process only one instance per repo (smoke test).",
    )
    args = parser.parse_args()

    import config as _cfg
    if args.test:
        _cfg.TEST_MODE = True

    target_repos = _resolve_repos(args.repos)

    if args.model:
        eval_dir = SUCCEEDED_DIR / args.model
        if not eval_dir.is_dir():
            raise SystemExit(f"Model directory not found: {eval_dir}")
        run_model(model=args.model, eval_dir=eval_dir, target_repos=target_repos)
    else:
        model_dirs = sorted(d for d in SUCCEEDED_DIR.iterdir() if d.is_dir())
        print(f"Found {len(model_dirs)} model(s): {[d.name for d in model_dirs]}")
        for model_dir in model_dirs:
            run_model(model=model_dir.name, eval_dir=model_dir, target_repos=target_repos)

    print("\nAll done.")


if __name__ == "__main__":
    main()
