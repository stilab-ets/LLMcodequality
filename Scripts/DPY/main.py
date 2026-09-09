"""
main.py — Entry point: load dataset → run pipeline → print summary.

Run from the project directory:
    python main.py
"""
from config             import OUT_DIR
from dataset_loader     import DatasetLoader
from instance_processor import InstanceProcessor


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    valid_instances = DatasetLoader().load()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    processor   = InstanceProcessor()
    all_results = []
    total       = len(valid_instances)

    print(f"\nRunning DPy pipeline on {total} instances")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 60)

    for i, inst in enumerate(valid_instances, 1):
        print(f"\n[{i}/{total}]")
        result = processor.process(inst)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    done    = sum(1 for r in all_results if r["status"] == "done")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    errors  = sum(1 for r in all_results if r["status"] == "error")

    print(f"\n{'=' * 60}")
    print(f"Done={done}  Skipped={skipped}  Errors={errors}")

    if errors:
        print("\nFailed instances:")
        for r in all_results:
            if r["status"] == "error":
                print(f"  {r['iid']}\n    {r['error']}")


if __name__ == "__main__":
    main()