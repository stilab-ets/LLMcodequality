"""
und_runner.py — Thin wrapper around the SciTools `und` executable.
"""
import os
import subprocess
from pathlib import Path

from config import SCI_BIN, UND_EXE


class UndRunner:
    """Runs `und` sub-commands and handles environment setup."""

    def __init__(self, sci_bin: str = SCI_BIN, und_exe: str = UND_EXE):
        self.sci_bin = sci_bin
        self.und_exe = und_exe
        assert Path(self.und_exe).exists(), f"und not found at: {self.und_exe}"

    def env(self) -> dict:
        """Return os.environ with LD_LIBRARY_PATH pointing at SciTools libs."""
        e = os.environ.copy()
        e["LD_LIBRARY_PATH"] = self.sci_bin + ":" + e.get("LD_LIBRARY_PATH", "")
        e["PERL5LIB"] = ""
        e["PERLLIB"]  = ""
        return e

    def run(self, args: list, timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Run an und sub-command.

        args: list of arguments AFTER the und executable,
              e.g. ["create", "-languages", "Python", str(udb_path)]
        """
        cmd   = [self.und_exe] + [str(a) for a in args]
        label = args[0] if args else "und"
        print(f"  [und {label}] {' '.join(str(a) for a in args[1:])[:120]}")

        result = subprocess.run(
            cmd, env=self.env(),
            text=True, capture_output=True, timeout=timeout,
        )
        if result.returncode == 0:
            print("  → OK")
        else:
            print(f"  → FAILED (rc={result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()[:300]}")
        return result