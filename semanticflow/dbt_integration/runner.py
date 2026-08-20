from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from semanticflow.dbt_integration._subprocess import (
    DEFAULT_COMMAND_TIMEOUT,
    RC_NOT_FOUND,
    RC_TIMEOUT,
    run_command,
)

# Default per-command timeout (seconds); a hung dbt invocation otherwise blocks the run.
DBT_COMMAND_TIMEOUT = DEFAULT_COMMAND_TIMEOUT


def _resolve_dbt_path() -> str:
    """Locate the dbt executable, preferring the active venv's bin/Scripts dir.

    Mirrors the MetricFlow runner's resolution so dbt is found even when the venv is not
    activated and `.venv/bin` is not on PATH (otherwise the bare name 'dbt' yields a
    127/command-not-found and every task fails parse)."""
    found = shutil.which("dbt")
    if found:
        return found
    bindir = Path(sys.executable).parent
    for candidate in (bindir / "dbt", bindir / "dbt.exe"):
        if candidate.exists():
            return str(candidate)
    return "dbt"  # last resort — will surface a clear 127 if truly missing


@dataclass
class DbCommandResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    error_types: list[str]

    @property
    def success(self) -> bool:
        return self.returncode == 0


def _parse_errors(output: str) -> list[str]:
    errors: set[str] = set()
    lowered = output.lower()
    if "yaml" in lowered and "error" in lowered:
        errors.add("invalid_yaml")
    if "column" in lowered and ("not found" in lowered or "does not exist" in lowered):
        errors.add("missing_column")
    if "time spine" in lowered:
        errors.add("missing_time_spine")
    if "ref" in lowered and ("was not found" in lowered or "not found" in lowered):
        errors.add("invalid_ref")
    if "compilation error" in lowered:
        errors.add("compilation_error")
    if "runtime error" in lowered:
        errors.add("runtime_error")
    if "was not found" in lowered and "model" in lowered:
        errors.add("missing_model")
    return sorted(errors)


def _run(
    command: list[str],
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = DBT_COMMAND_TIMEOUT,
) -> DbCommandResult:
    raw = run_command(command, env=env, cwd=cwd, timeout=timeout)
    if raw.returncode == RC_TIMEOUT:
        error_types = ["timeout"]
    elif raw.returncode == RC_NOT_FOUND:
        error_types = ["command_not_found"]
    else:
        error_types = _parse_errors(f"{raw.stdout}\n{raw.stderr}".strip())
    return DbCommandResult(
        command=raw.command,
        stdout=raw.stdout,
        stderr=raw.stderr,
        returncode=raw.returncode,
        error_types=error_types,
    )


def _apply_profile_args(
    command: list[str],
    profile: str | None = None,
    profiles_dir: str | None = None,
) -> list[str]:
    if profile:
        command.extend(["--profile", profile])
    if profiles_dir:
        resolved_profiles = str(Path(profiles_dir).resolve())
        command.extend(["--profiles-dir", resolved_profiles])
    return command


def run_dbt_parse(
    project_dir: str,
    env: dict[str, str] | None = None,
    profile: str | None = None,
    profiles_dir: str | None = None,
) -> DbCommandResult:
    resolved = str(Path(project_dir).resolve())
    command = [_resolve_dbt_path(), "parse", "--project-dir", resolved]
    command = _apply_profile_args(command, profile, profiles_dir)
    return _run(command, env=env, cwd=resolved)


def run_dbt_seed(
    project_dir: str,
    env: dict[str, str] | None = None,
    profile: str | None = None,
    profiles_dir: str | None = None,
) -> DbCommandResult:
    resolved = str(Path(project_dir).resolve())
    command = [_resolve_dbt_path(), "seed", "--project-dir", resolved]
    command = _apply_profile_args(command, profile, profiles_dir)
    return _run(command, env=env, cwd=resolved)


def run_dbt_build(
    project_dir: str,
    env: dict[str, str] | None = None,
    select: Iterable[str] | None = None,
    profile: str | None = None,
    profiles_dir: str | None = None,
) -> DbCommandResult:
    resolved = str(Path(project_dir).resolve())
    command = [_resolve_dbt_path(), "build", "--project-dir", resolved]
    if select:
        command.append("--select")
        command.extend(select)
    command = _apply_profile_args(command, profile, profiles_dir)
    return _run(command, env=env, cwd=resolved)
