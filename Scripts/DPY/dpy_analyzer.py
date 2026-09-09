"""
dpy_analyzer.py — Load, filter, and save DPy output CSVs.
"""
from pathlib import Path

import pandas as pd

from config import (
    DPY_ALL_FILES, DPY_FILE_COLUMN,
)


class DpyAnalyzer:
    """Loads DPy output CSVs, filters them to changed files, and saves results."""

    def get_filename(self, out_dir: Path, suffix: str) -> Path:
        """
        DPy prefixes output files with the input folder name.
        e.g. input folder = 'ansible' → 'ansible_arch_smells.csv'
        Find the file by suffix regardless of prefix.
        """
        matches = list(out_dir.glob(f"*_{suffix}"))
        if matches:
            return matches[0]
        return out_dir / suffix

    def load_csvs(self, dpy_out_dir: Path) -> dict[str, pd.DataFrame]:
        """
        Load all DPy output CSVs from a directory.
        Keys are suffix names (e.g. 'arch_smells.csv').
        Missing files are returned as empty DataFrames.
        """
        dfs = {}
        for fname in DPY_ALL_FILES:
            fpath = self.get_filename(dpy_out_dir, fname)
            if fpath.exists() and fpath.stat().st_size > 0:
                dfs[fname] = pd.read_csv(fpath, encoding="utf-8", encoding_errors="replace")
            else:
                dfs[fname] = pd.DataFrame()
        return dfs

    def filter_to_changed(
        self,
        dfs:          dict[str, pd.DataFrame],
        changed_files: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Filter DPy DataFrames to only rows belonging to changed files.
        DPy stores the source file stem in the 'Module' column.
        e.g. 'ansible/vars/manager.py' → Module = 'manager'
        """
        tail_set   = {p.replace("\\", "/") for p in changed_files}
        module_set = {Path(p).stem for p in changed_files}

        def matches_module(val: str) -> bool:
            v = str(val).strip()
            if v in module_set:
                return True
            v_fwd = v.replace("\\", "/")
            return any(
                v_fwd == t or v_fwd.endswith("/" + t) or v_fwd.endswith(t)
                for t in tail_set
            )

        filtered = {}
        for fname, df in dfs.items():
            if df.empty:
                filtered[fname] = df
                continue
            col = DPY_FILE_COLUMN.get(fname)
            if col is None:
                # arch_smells — package level, keep all
                filtered[fname] = df.copy()
            elif col in df.columns:
                mask = df[col].apply(matches_module)
                filtered[fname] = df[mask].reset_index(drop=True)
            else:
                print(f"  [warn] {fname} missing column '{col}' — cols: {list(df.columns)}")
                filtered[fname] = df.copy()
        return filtered

    def load_and_concat_csvs(self, dpy_out_dirs: list[Path]) -> dict[str, pd.DataFrame]:
        """Load DPy CSVs from multiple output directories and concatenate them."""
        combined: dict[str, list[pd.DataFrame]] = {f: [] for f in DPY_ALL_FILES}
        for d in dpy_out_dirs:
            for fname, df in self.load_csvs(d).items():
                if not df.empty:
                    combined[fname].append(df)
        return {
            fname: (
                pd.concat(dfs, ignore_index=True).drop_duplicates()
                if dfs else pd.DataFrame()
            )
            for fname, dfs in combined.items()
        }

    def save_csvs(
        self,
        dfs:     dict[str, pd.DataFrame],
        out_dir: Path,
        prefix:  str,
    ) -> None:
        """Save DataFrames with a given prefix (e.g. 'before_gold', 'after_llm')."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, df in dfs.items():
            stem     = Path(fname).stem
            out_path = out_dir / f"{prefix}_{stem}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  [saved] {out_path.name}: {len(df)} rows")