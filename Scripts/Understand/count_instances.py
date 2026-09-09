"""Count succeeded instances per model in python_data/succeeded."""
from pathlib import Path

SUCCEEDED_DIR = Path("/home/AV00500/Issam/SWE-Bench_Pro/swebench_pro_pipeline/python_data/succeeded")

def main():
    models = sorted(d for d in SUCCEEDED_DIR.iterdir() if d.is_dir())
    total = 0
    print(f"{'Model':<35} {'Instances':>10}")
    print("-" * 47)
    for model_dir in models:
        count = sum(1 for d in model_dir.iterdir() if d.is_dir())
        print(f"{model_dir.name:<35} {count:>10}")
        total += count
    print("-" * 47)
    print(f"{'TOTAL':<35} {total:>10}")

if __name__ == "__main__":
    main()
