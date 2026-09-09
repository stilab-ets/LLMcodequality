"""
unique_symbols.py — Scan all messages_diff_*.csv files across all models/instances
and write a CSV of unique pylint symbols with occurrence counts and an example message.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "pylint_results"
OUT_CSV = Path(__file__).parent / "unique_symbols.csv"


def main() -> None:
    symbol_total: dict[str, int] = defaultdict(int)
    symbol_instances: dict[str, set] = defaultdict(set)
    symbol_example: dict[str, str] = {}

    files = sorted(RESULTS_DIR.rglob("messages_diff_*.csv"))
    if not files:
        print("No messages_diff_*.csv files found under", RESULTS_DIR)
        sys.exit(1)

    print(f"Scanning {len(files)} messages_diff_*.csv files...")

    for csv_path in files:
        parts = csv_path.relative_to(RESULTS_DIR).parts
        model = parts[0] if len(parts) > 0 else "unknown"
        instance = parts[2] if len(parts) > 2 else csv_path.parent.name

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] could not read {csv_path}: {e}")
            continue

        if "symbol" not in df.columns:
            continue

        has_message = "message" in df.columns

        for _, row in df.iterrows():
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            symbol_total[sym] += 1
            symbol_instances[sym].add((model, instance))
            if sym not in symbol_example and has_message:
                msg = str(row.get("message", "")).strip()
                if msg and msg.lower() != "nan":
                    symbol_example[sym] = msg

    if not symbol_total:
        print("No symbols found.")
        return

    rows = [
        {
            "symbol": sym,
            "occurrences": symbol_total[sym],
            "n_instances": len(symbol_instances[sym]),
            "n_models": len({m for m, _ in symbol_instances[sym]}),
            "example_message": symbol_example.get(sym, ""),
        }
        for sym in sorted(symbol_total.keys())
    ]

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Written {len(rows)} unique symbols → {OUT_CSV}")


if __name__ == "__main__":
    main()
