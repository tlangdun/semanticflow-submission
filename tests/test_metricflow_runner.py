"""Test MetricFlow runner path resolution fix."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semanticflow.dbt_integration.metricflow_runner import (
    _retry_without_project_flags,
    run_mf_validate,
    MfCommandResult,
)
from semanticflow.config import Settings


def test_retry_without_project_flags_sets_both_env_vars():
    """Test that _retry_without_project_flags sets both DBT_PROFILES_DIR and DBT_PROJECT_DIR."""
    
    # Simulate a failed result that triggers the retry
    failed_result = MfCommandResult(
        command=["mf", "validate-configs", "--project-dir", "third_party/jaffle_shop_duckdb"],
        stdout="",
        stderr="Missing: 'third_party/jaffle_shop_duckdb/dbt_project.yml'",
        returncode=1,
        issues=[{"message": "Missing dbt_project.yml"}],
    )
    
    project_dir = "third_party/jaffle_shop_duckdb"
    expected_abs_path = str(Path(project_dir).resolve())
    
    captured_env = {}
    captured_cwd = None
    
    def mock_run(command, env=None, cwd=None):
        nonlocal captured_env, captured_cwd
        captured_env = env or {}
        captured_cwd = cwd
        return MfCommandResult(
            command=command,
            stdout="OK",
            stderr="",
            returncode=0,
            issues=[],
        )
    
    with patch("semanticflow.dbt_integration.metricflow_runner._run", side_effect=mock_run):
        result, command = _retry_without_project_flags(
            failed_result,
            failed_result.command,
            project_dir,
            env={},
        )
    
    # Verify both env vars are set
    assert "DBT_PROFILES_DIR" in captured_env, "DBT_PROFILES_DIR should be set"
    assert "DBT_PROJECT_DIR" in captured_env, "DBT_PROJECT_DIR should be set"
    
    # Verify they use absolute paths
    assert captured_env["DBT_PROFILES_DIR"] == expected_abs_path, \
        f"DBT_PROFILES_DIR should be absolute: {captured_env['DBT_PROFILES_DIR']} != {expected_abs_path}"
    assert captured_env["DBT_PROJECT_DIR"] == expected_abs_path, \
        f"DBT_PROJECT_DIR should be absolute: {captured_env['DBT_PROJECT_DIR']} != {expected_abs_path}"
    
    # Verify cwd is also absolute
    assert captured_cwd == expected_abs_path, \
        f"cwd should be absolute: {captured_cwd} != {expected_abs_path}"
    
    print("✅ _retry_without_project_flags correctly sets both env vars with absolute paths")
    return True


def test_mf_validate_integration():
    """Integration test: run actual mf validate against jaffle_shop."""
    
    project_root = Path(__file__).parent.parent
    jaffle_shop_dir = project_root / "third_party" / "jaffle_shop_duckdb"
    
    if not jaffle_shop_dir.exists():
        print(f"⚠️  Skipping integration test: {jaffle_shop_dir} not found")
        return True
    
    # Load settings from env
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
    except ImportError:
        # dotenv not available, assume env vars are already set
        pass
    
    settings = Settings()
    
    print(f"Running mf validate against: {jaffle_shop_dir}")
    result = run_mf_validate(str(jaffle_shop_dir), settings)
    
    print(f"Command: {' '.join(result.command)}")
    print(f"Return code: {result.returncode}")
    print(f"Success: {result.success}")
    
    if result.stdout:
        print(f"Stdout:\n{result.stdout}")
    if result.stderr:
        print(f"Stderr:\n{result.stderr}")
    
    # Check for the specific path resolution error we fixed
    path_error = "Missing:" in (result.stderr or "") and "dbt_project.yml" in (result.stderr or "")
    
    if path_error:
        print("❌ Path resolution issue still present!")
        return False
    
    # Check if manifest was parsed successfully (our fix worked)
    manifest_parsed = "Successfully parsed manifest" in (result.stdout or "")
    
    if manifest_parsed:
        print("✅ Path resolution fix verified - manifest parsed successfully!")
        if not result.success:
            print("   ℹ️  Note: mf validate returned errors unrelated to path resolution")
            print("   ℹ️  (e.g., invalid metric filters from a previous task run)")
        return True
    
    if result.success:
        print("✅ mf validate succeeded!")
        return True
    else:
        print("❌ mf validate failed for unknown reason")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MetricFlow Runner Path Resolution Fix")
    print("=" * 60)
    
    # Unit test
    print("\n[1/2] Unit test: _retry_without_project_flags")
    unit_ok = test_retry_without_project_flags_sets_both_env_vars()
    
    # Integration test
    print("\n[2/2] Integration test: run_mf_validate")
    integration_ok = test_mf_validate_integration()
    
    print("\n" + "=" * 60)
    if unit_ok and integration_ok:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
