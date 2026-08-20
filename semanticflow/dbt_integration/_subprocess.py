from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

# Default per-command timeout (seconds); a hung dbt/mf invocation otherwise blocks the run.
DEFAULT_COMMAND_TIMEOUT = 600

# Sentinel return codes for failures that never reach the child process.
RC_TIMEOUT = 124
RC_NOT_FOUND = 127


@dataclass
class RawResult:
    """Provider-agnostic outcome of a subprocess invocation. The dbt/mf runners wrap
    this with their own result dataclasses + output post-processing."""
    command: list[str]
    stdout: str
    stderr: str
    returncode: int


def run_command(
    command: list[str],
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    use_shell: bool = False,
) -> RawResult:
    """Run a command capturing stdout/stderr as UTF-8 (replacing undecodable bytes).

    Starts from the full environment so PATH/HOME survive a custom ``env``, then merges
    overrides and forces UTF-8 (needed on Windows). A timeout returns RC_TIMEOUT and a
    missing executable returns RC_NOT_FOUND instead of raising, so every caller gets a
    result object rather than an exception.

    ``use_shell`` only takes effect on win32, where ``mf`` needs shell=True to dodge
    WinError 10106 (Winsock init failure).
    """
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    run_env["PYTHONUTF8"] = "1"
    run_env["PYTHONIOENCODING"] = "utf-8"

    kwargs: dict = dict(
        capture_output=True,
        text=True,
        env=run_env,
        cwd=cwd,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    try:
        if use_shell and sys.platform == "win32":
            # list2cmdline quotes paths with spaces for the Windows shell.
            result = subprocess.run(subprocess.list2cmdline(command), shell=True, **kwargs)
        else:
            result = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as exc:
        return RawResult(command, exc.stdout or "", f"command timed out after {timeout}s: {exc}", RC_TIMEOUT)
    except FileNotFoundError as exc:
        return RawResult(command, "", str(exc), RC_NOT_FOUND)
    return RawResult(command, result.stdout, result.stderr, result.returncode)
