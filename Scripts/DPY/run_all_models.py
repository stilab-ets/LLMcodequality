"""
run_all_models.py — Run the DPy pipeline over every model in python_data/succeeded.

Skips instances that are already processed (handled by InstanceProcessor).

Usage:
    python run_all_models.py
"""
from pathlib import Path

from dataset_loader     import DatasetLoader
from instance_processor import InstanceProcessor
from config             import REPO_MAP, REPO_LANG, TARGET_INSTANCE, TARGET_REPO

SUCCEEDED_ROOT = Path(r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\python_data\succeeded")
OUT_ROOT       = Path("dpy_results")


def main():
    models = sorted(p.name for p in SUCCEEDED_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(models)} model(s): {models}\n")

    grand_done = grand_skipped = grand_errors = 0

    for model in models:
        eval_dir = SUCCEEDED_ROOT / model
        out_dir  = OUT_ROOT / model
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model}")
        print(f"{'#' * 60}")

        valid_instances = DatasetLoader(
            eval_dir        = eval_dir,
            repo_map        = REPO_MAP,
            repo_lang       = REPO_LANG,
            target_instance = TARGET_INSTANCE,
            target_repo     = TARGET_REPO,
        ).load()

        if not valid_instances:
            print("  No valid instances — skipping model.")
            continue

        processor   = InstanceProcessor(out_dir=out_dir)
        all_results = []
        total       = len(valid_instances)

        print(f"\nRunning DPy pipeline on {total} instances")
        print(f"  Output dir: {out_dir}")
        print("=" * 60)

        for i, inst in enumerate(valid_instances, 1):
            print(f"\n[{i}/{total}]")
            result = processor.process(inst)
            all_results.append(result)

        done    = sum(1 for r in all_results if r["status"] == "done")
        skipped = sum(1 for r in all_results if r["status"] == "skipped")
        errors  = sum(1 for r in all_results if r["status"] == "error")

        print(f"\n{'=' * 60}")
        print(f"[{model}] Done={done}  Skipped={skipped}  Errors={errors}")

        if errors:
            print("  Failed instances:")
            for r in all_results:
                if r["status"] == "error":
                    print(f"    {r['iid']}\n      {r['error']}")

        grand_done    += done
        grand_skipped += skipped
        grand_errors  += errors

    print(f"\n{'#' * 60}")
    print(f"GRAND TOTAL — Done={grand_done}  Skipped={grand_skipped}  Errors={grand_errors}")


if __name__ == "__main__":
    main()
