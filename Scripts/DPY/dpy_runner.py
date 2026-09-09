"""
dpy_runner.py — Thin wrapper around the DPy executable.
"""
import subprocess
from pathlib import Path

from config import DPY_EXE


class DpyRunner:
    """Runs DPy sub-commands."""

    def __init__(self, dpy_exe: str = DPY_EXE):
        self.dpy_exe = dpy_exe
        assert Path(self.dpy_exe).exists(), f"DPy not found at: {self.dpy_exe}"

    def run(self, args: list, timeout: int = 600) -> subprocess.CompletedProcess:
        """Run a DPy sub-command and print the step."""
        cmd   = [str(self.dpy_exe)] + [str(a) for a in args]
        label = args[0] if args else "dpy"
        print(f"  [dpy {label}] {' '.join(str(a) for a in args[1:])[:120]}")

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            shell=True,   # needed on Windows for exe paths with spaces
        )
        if result.returncode == 0:
            print("  → OK")
        else:
            print(f"  → FAILED (rc={result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()[:300]}")
        return result

    def analyze(self, input_path: Path, output_path: Path) -> bool:
        """
        Run DPy analysis on input_path, store CSVs in output_path.
        Returns True on success.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  [dpy analyze] {input_path.name} → {output_path.name}")
        result = self.run(
            ["analyze", "-i", str(input_path), "-o", str(output_path), "-f", "csv"],
        )
        return result.returncode == 0