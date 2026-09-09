"""
dataset_loader.py — Load the SWE-Bench Pro dataset and collect valid instances.
"""
import re
from pathlib import Path

import pandas as pd

from config import EVAL_DIR, REPO_MAP, REPO_LANG, TARGET_INSTANCE, TARGET_REPO


class DatasetLoader:
    """Loads the SWE-Bench Pro parquet and collects valid instance folders."""

    _INST_RE    = re.compile(r"^instance_(.+)-[0-9a-f]{40}-v", re.I)
    _DATASET_URL = "hf://datasets/ScaleAI/SWE-bench_Pro/data/test-00000-of-00001.parquet"

    def __init__(
        self,
        eval_dir:        Path = EVAL_DIR,
        repo_map:        dict = REPO_MAP,
        repo_lang:       dict = REPO_LANG,
        target_instance: str  = TARGET_INSTANCE,
        target_repo:     str  = TARGET_REPO,
    ):
        self.eval_dir        = eval_dir
        self.repo_map        = repo_map
        self.repo_lang       = repo_lang
        self.target_instance = target_instance
        self.target_repo     = target_repo

    def load(self) -> list[dict]:
        """
        Load the dataset and return a flat list of valid instance dicts.

        Each dict contains:
            inst_dir, iid, repo_id, repo_name, lang,
            base_commit, gold_patch_text, llm_patch_path
        """
        print("Loading dataset...")
        df = pd.read_parquet(self._DATASET_URL)
        commit_map = dict(zip(df["instance_id"], df["base_commit"]))
        patch_map  = dict(zip(df["instance_id"], df["patch"]))
        print(f"  → {len(commit_map)} instances loaded from dataset")

        assert self.eval_dir.is_dir(), f"Eval dir not found: {self.eval_dir}"
        folders = list(self.eval_dir.iterdir())
        if self.target_instance:
            folders = [p for p in folders if p.name == self.target_instance]
        print(f"  → {len(folders)} instance folders found in eval dir")

        valid = []
        for inst_dir in folders:
            if not inst_dir.is_dir():
                continue
            m = self._INST_RE.match(inst_dir.name)
            if not m:
                continue
            repo_id = m.group(1)
            if repo_id not in self.repo_map:
                continue
            if self.target_repo and repo_id != self.target_repo:
                continue
            iid         = inst_dir.name
            base_commit = commit_map.get(iid)
            if not base_commit:
                continue
            valid.append({
                "inst_dir":        inst_dir,
                "iid":             iid,
                "repo_id":         repo_id,
                "repo_name":       self.repo_map[repo_id],
                "lang":            self.repo_lang[repo_id],
                "base_commit":     base_commit,
                "gold_patch_text": patch_map.get(iid, ""),
                "llm_patch_path":  inst_dir / "_patch.diff",
            })

        print(f"  → {len(valid)} valid instances after filtering")
        for inst in valid[:5]:
            print(f"  {inst['iid']}  |  repo={inst['repo_name']}  |  commit={inst['base_commit'][:10]}...")
        return valid