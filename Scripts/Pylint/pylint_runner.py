"""
pylint_runner.py — Thin wrapper around the pylint CLI that returns parsed JSON messages.
"""
import json
import subprocess
from pathlib import Path

from config import PYLINT_EXTRA_ARGS


class PylintRunner:
    """
    Runs pylint on a list of files and returns the raw list of message dicts.

    Each message dict has the keys pylint emits in JSON mode:
        type, module, obj, line, column, endLine, endColumn,
        path, symbol, message, message-id
    """

    def run(
        self,
        files: list[str],
        cwd: Path,
        timeout: int = 300,
    ) -> list[dict]:
        """
        Run pylint on *files* (relative paths from *cwd*) and return the
        parsed JSON message list.  Returns [] on timeout or fatal crash.
        """
        if not files:
            return []

        cmd = ["pylint"] + PYLINT_EXTRA_ARGS + list(files)
        print(f"  [pylint] running on {len(files)} file(s)...")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  [pylint] TIMEOUT after {timeout}s — returning empty")
            return []

        # pylint exit codes: 0=ok, 1=fatal, 2=error, 4=warning, 8=refactor,
        # 16=convention, 32=usage error. Any combination is OR'd together.
        # Exit code 32 means a usage error (bad file path etc.), treat as fatal.
        if result.returncode == 32:
            print(f"  [pylint] usage error:\n{result.stderr[:400]}")
            return []

        stdout = result.stdout.strip()
        if not stdout:
            return []

        try:
            messages = json.loads(stdout)
        except json.JSONDecodeError:
            # pylint may prefix non-JSON lines (e.g. "No config file found")
            # Try to isolate the JSON array.
            start = stdout.find("[")
            end   = stdout.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    messages = json.loads(stdout[start:end])
                except json.JSONDecodeError:
                    print(f"  [pylint] could not parse JSON output — returning empty")
                    return []
            else:
                print(f"  [pylint] no JSON array found in output — returning empty")
                return []

        print(f"  [pylint] {len(messages)} messages")
        return messages
