"""
categorize_patch_size.py — Classify SWE-Bench Pro instances into Small / Medium / Large
based on gold patch size (lines added + deleted).

Gold patches are read from the HuggingFace dataset:
  hf://datasets/ScaleAI/SWE-bench_Pro/data/test-00000-of-00001.parquet  (column: 'patch')

Repo is extracted from the instance_id field using the naming scheme:
  instance_{owner}__{repo}-{commit_hash}-v{version_hash}

Thresholds derived from the gold patch distribution:
  - Small:  size <= Q1
  - Medium: Q1 < size <= Q3
  - Large:  size > Q3

Outputs (written to understand_results/):
  patch_size_lookup.csv   -- instance_id, repo, gold_patch_lines, size_category, Q1, Q3
  patch_size_stats.txt    -- summary statistics
"""

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

DATASET_URL  = "hf://datasets/ScaleAI/SWE-bench_Pro/data/test-00000-of-00001.parquet"
RESULTS_ROOT = Path(r"C:\Users\AV00500\Desktop\Project\swebench_pro_pipeline\understand_results")
RESULTS_ROOT.mkdir(exist_ok=True)

_INST_RE = re.compile(r"^instance_(.+)-[0-9a-f]{40}-v", re.I)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def extract_repo(instance_id: str) -> str:
    """Extract '{owner}__{repo}' from an instance_id string."""
    m = _INST_RE.match(instance_id)
    return m.group(1) if m else instance_id


def count_patch_lines(patch_text) -> int:
    """Count lines added + lines deleted in a unified diff."""
    if not patch_text or not isinstance(patch_text, str):
        return 0
    added = deleted = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return added + deleted


def quantile(sorted_values: list, q: float) -> float:
    """Linear-interpolation quantile (matches numpy/pandas default)."""
    n = len(sorted_values)
    pos = q * (n - 1)
    lo = int(pos)
    hi = lo + 1
    frac = pos - lo
    if hi >= n:
        return float(sorted_values[lo])
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"Loading parquet from {DATASET_URL} ...")
    df = pd.read_parquet(DATASET_URL)
    print(f"  {len(df)} instances, columns: {df.columns.tolist()}")

    instances = []
    skipped = 0

    for _, row in df.iterrows():
        instance_id = row["instance_id"]
        repo        = extract_repo(instance_id)
        patch_text  = row.get("patch", "")
        size        = count_patch_lines(patch_text)

        if size == 0:
            skipped += 1

        instances.append({
            "instance_id":      instance_id,
            "repo":             repo,
            "gold_patch_lines": size,
        })

    print(f"  Processed {len(instances)} instances ({skipped} with zero-size patch)")

    # -- Compute thresholds ------------------------------------------
    sizes = sorted(inst["gold_patch_lines"] for inst in instances)
    q1   = quantile(sizes, 0.25)
    q2   = quantile(sizes, 0.50)
    q3   = quantile(sizes, 0.75)
    iqr  = q3 - q1
    mean = sum(sizes) / len(sizes)

    print(f"\n  Gold patch size distribution (lines added + deleted):")
    print(f"    Min:    {sizes[0]}")
    print(f"    Q1:     {q1}")
    print(f"    Median: {q2}")
    print(f"    Q3:     {q3}")
    print(f"    Max:    {sizes[-1]}")
    print(f"    IQR:    {iqr}")
    print(f"    Mean:   {mean:.1f}")

    # -- Categorize --------------------------------------------------
    for inst in instances:
        s = inst["gold_patch_lines"]
        if s <= q1:
            inst["size_category"] = "Small"
        elif s <= q3:
            inst["size_category"] = "Medium"
        else:
            inst["size_category"] = "Large"

    cat_counts = Counter(inst["size_category"] for inst in instances)
    print(f"\n  Categories:")
    print(f"    Small  (<= {q1}):          {cat_counts['Small']}")
    print(f"    Medium ({q1} < x <= {q3}): {cat_counts['Medium']}")
    print(f"    Large  (> {q3}):           {cat_counts['Large']}")

    # -- Per-repo breakdown ------------------------------------------
    repo_cats: dict[str, Counter] = {}
    for inst in instances:
        repo_cats.setdefault(inst["repo"], Counter())[inst["size_category"]] += 1

    print(f"\n  Per-repo breakdown:")
    for repo in sorted(repo_cats):
        counts = repo_cats[repo]
        total  = sum(counts.values())
        print(f"    {repo}: S={counts.get('Small',0)} M={counts.get('Medium',0)} L={counts.get('Large',0)} (total={total})")

    # -- Write lookup CSV --------------------------------------------
    lookup_csv = RESULTS_ROOT / "patch_size_lookup.csv"
    fieldnames = ["instance_id", "repo", "gold_patch_lines", "size_category", "Q1", "Q3"]
    with open(lookup_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for inst in sorted(instances, key=lambda x: x["instance_id"]):
            writer.writerow({
                "instance_id":      inst["instance_id"],
                "repo":             inst["repo"],
                "gold_patch_lines": inst["gold_patch_lines"],
                "size_category":    inst["size_category"],
                "Q1":               q1,
                "Q3":               q3,
            })
    print(f"\n  Lookup CSV -> {lookup_csv}  ({len(instances)} rows)")

    # -- Write stats file --------------------------------------------
    stats_file = RESULTS_ROOT / "patch_size_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("Gold Patch Size Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source: {DATASET_URL}\n")
        f.write(f"Total instances: {len(instances)}\n\n")
        f.write("Distribution (lines added + deleted):\n")
        f.write(f"  Min:    {sizes[0]}\n")
        f.write(f"  Q1:     {q1}\n")
        f.write(f"  Median: {q2}\n")
        f.write(f"  Q3:     {q3}\n")
        f.write(f"  Max:    {sizes[-1]}\n")
        f.write(f"  IQR:    {iqr}\n")
        f.write(f"  Mean:   {mean:.1f}\n\n")
        f.write("Thresholds:\n")
        f.write(f"  Small:  gold_patch_lines <= {q1}\n")
        f.write(f"  Medium: {q1} < gold_patch_lines <= {q3}\n")
        f.write(f"  Large:  gold_patch_lines > {q3}\n\n")
        f.write("Category counts:\n")
        f.write(f"  Small:  {cat_counts['Small']}\n")
        f.write(f"  Medium: {cat_counts['Medium']}\n")
        f.write(f"  Large:  {cat_counts['Large']}\n\n")
        f.write("Per-repo breakdown:\n")
        for repo in sorted(repo_cats):
            counts = repo_cats[repo]
            total  = sum(counts.values())
            f.write(f"  {repo}: S={counts.get('Small',0)} M={counts.get('Medium',0)} L={counts.get('Large',0)} (total={total})\n")

    print(f"  Stats      -> {stats_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
