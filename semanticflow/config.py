from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    """Load .env file into os.environ if it exists."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle 'export VAR=value' format
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env file on module import
_load_dotenv()


@dataclass
class Settings:
    dbt_project_dir: str = "third_party/jaffle_shop_duckdb"
    dbt_profile_name: str | None = None
    duckdb_path: str | None = None

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-2.0-flash"
    openai_base_url: str | None = None

    # OpenRouter: route every provider through the OpenAI-compatible OpenRouter API to avoid
    # per-provider rate limits. Model names stay the same and are mapped to OpenRouter slugs
    # ("<vendor>/<model>"); a model already containing "/" is passed through unchanged.
    openrouter_api_key: str | None = None
    use_openrouter: bool = False

    llm_provider: str = "mock"
    # Providers forming the multi-agent ensemble (the cross-model epistemic signal).
    # Defaults to all three; set to a single provider (e.g. ["anthropic"]) to run the
    # single-provider hybrid-UQ configuration (aleatoric + structural only, no epistemic).
    ensemble_providers: list[str] = field(
        default_factory=lambda: ["openai", "anthropic", "gemini"]
    )

    mode: str = "multi_agent_no_hitl"
    agreement_threshold: float = 0.75
    uncertainty_threshold: float = 0.5
    hitl_answer_budget: int = 3  # max questions the simulated human will answer per task
    # Simulated-human mode for HITL experiments: "keyword" (deterministic slot-scoped
    # reveal, zero API cost — the historical behavior) or "llm" (a cheap-model human
    # conditioned on the hidden gold intent answering in free-form NL, which removes the
    # canned-phrasing handshake between sim answers and the deterministic refiner).
    sim_human_mode: str = "keyword"
    # Cheap-tier client used for low-stakes calls (sim-human answers, last-resort
    # execution repair, clarifying-question generation). None = derive from llm_provider/
    # ensemble_providers with a per-provider cheap default model (e.g. haiku for anthropic).
    cheap_llm_provider: str | None = None
    cheap_llm_model: str | None = None
    # Allow ONE cheap-LLM repair call per task, only after deterministic execution-repair
    # found nothing and the query still fails. Costs nothing on tasks that execute.
    llm_execution_repair: bool = True
    # In HITL mode, ask the (simulated) human one deterministic question generated from
    # the MetricFlow error when the query still fails after all repair loops.
    post_execution_hitl: bool = True
    # Slot-scoped HITL refinement guard: after a HITL refinement, revert any slot the
    # refinement OVERWROTE that all ensemble providers had already agreed on (non-empty +
    # unanimous) — these "protected overwrites" harm already-correct tasks. Set False to
    # ablate (full refinement, no guard). No-op for single-provider runs (< 2 providers).
    slot_scoped_hitl: bool = True
    # Use an LLM to incorporate clarifying answers into the spec (robust to free-form,
    # natural-language answers), instead of the deterministic content parser which expects
    # canonical phrasing. The deterministic parser still runs first; the LLM path is invoked
    # whenever answers are present, and its output is preferred when it changed the spec.
    llm_refiner: bool = False
    self_consistency_k: int = 1  # samples per provider for aleatoric uncertainty (1 = off)
    # Execution-based consistency (the strongest black-box UQ signal for text-to-SQL):
    # when > 0, each provider's representative spec is executed and the result tables are
    # clustered by execution-equivalence to derive `execution_uncertainty`. 0 = off (the
    # extra dbt/mf runs are costly). See evaluation/execution_consistency.py.
    execution_consistency: bool = False
    experiment_repeats: int = 1  # full-pipeline repeats per task to quantify LLM stochasticity
    # Hard wall-clock cap (seconds) per task in run_experiment: if a task's repeats loop
    # exceeds it (e.g. a stalled LLM call with no inner timeout), the task is aborted and
    # recorded as a timeout stub instead of hanging the whole run. 0 = disabled.
    task_timeout_sec: int = 0
    run_dbt: bool = True
    max_llm_repair_attempts: int = 1
    llm_timeout_sec: float = 60.0
    llm_repair_timeout_sec: float = 5.0
    llm_max_retries: int = 1
    llm_max_tokens: int = 4096  # cap on LLM output tokens; too small silently truncates JSON specs
    log_llm_raw: bool = False
    # Disk-cache LLM responses to cut API spend on repeated dev runs. OFF by default and
    # FORCE-DISABLED when self_consistency_k > 1 (identical cached responses would collapse
    # the aleatoric signal to zero). Safe with K=1. See semanticflow/llm/cache.py.
    llm_cache: bool = False

    mf_cli_path: str = "mf"
    mf_cli_mode: str = "mf"
    mf_validate_warehouse: bool = False
    mf_query_output: str = "json"
    skip_mf_validation: bool = False  # Skip MF validation (use when MF CLI has issues)
    skip_mf_query: bool = False  # Skip MF query (use when MF CLI has issues on Windows)
    enable_duckdb_fallback: bool = False  # raw-SQL fallback when mf fails; OFF — it ignores
                                          # filters and can return silently-wrong results

    # Metric naming options
    normalize_metric_names: bool = True  # Normalize metric names to snake_case
    metric_name_prefix: str | None = None  # Optional prefix for metric names
    
    # Logging options
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_format: str = "text"  # text or json

    config_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_ENV_MAP = {
    "DBT_PROJECT_DIR": "dbt_project_dir",
    "DBT_PROFILE_NAME": "dbt_profile_name",
    "DUCKDB_PATH": "duckdb_path",
    "OPENAI_API_KEY": "openai_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "GEMINI_API_KEY": "gemini_api_key",
    "OPENAI_MODEL": "openai_model",
    "ANTHROPIC_MODEL": "anthropic_model",
    "GEMINI_MODEL": "gemini_model",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENROUTER_API": "openrouter_api_key",
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "SEMANTICFLOW_USE_OPENROUTER": "use_openrouter",
    "LLM_PROVIDER": "llm_provider",
    "SEMANTICFLOW_ENSEMBLE_PROVIDERS": "ensemble_providers",
    "SEMANTICFLOW_MODE": "mode",
    "SEMANTICFLOW_AGREEMENT_THRESHOLD": "agreement_threshold",
    "SEMANTICFLOW_UNCERTAINTY_THRESHOLD": "uncertainty_threshold",
    "SEMANTICFLOW_HITL_ANSWER_BUDGET": "hitl_answer_budget",
    "SEMANTICFLOW_SIM_HUMAN": "sim_human_mode",
    "SEMANTICFLOW_CHEAP_PROVIDER": "cheap_llm_provider",
    "SEMANTICFLOW_CHEAP_MODEL": "cheap_llm_model",
    "SEMANTICFLOW_LLM_EXECUTION_REPAIR": "llm_execution_repair",
    "SEMANTICFLOW_POST_EXEC_HITL": "post_execution_hitl",
    "SEMANTICFLOW_SLOT_SCOPED_HITL": "slot_scoped_hitl",
    "SEMANTICFLOW_LLM_REFINER": "llm_refiner",
    "SEMANTICFLOW_SELF_CONSISTENCY_K": "self_consistency_k",
    "SEMANTICFLOW_EXECUTION_CONSISTENCY": "execution_consistency",
    "SEMANTICFLOW_EXPERIMENT_REPEATS": "experiment_repeats",
    "SEMANTICFLOW_TASK_TIMEOUT_SEC": "task_timeout_sec",
    "SEMANTICFLOW_RUN_DBT": "run_dbt",
    "SEMANTICFLOW_MAX_LLM_REPAIR": "max_llm_repair_attempts",
    "SEMANTICFLOW_LLM_TIMEOUT_SEC": "llm_timeout_sec",
    "SEMANTICFLOW_LLM_REPAIR_TIMEOUT_SEC": "llm_repair_timeout_sec",
    "SEMANTICFLOW_LLM_MAX_RETRIES": "llm_max_retries",
    "SEMANTICFLOW_LLM_MAX_TOKENS": "llm_max_tokens",
    "SEMANTICFLOW_LOG_LLM_RAW": "log_llm_raw",
    "SEMANTICFLOW_LLM_CACHE": "llm_cache",
    "MF_CLI_PATH": "mf_cli_path",
    "MF_CLI_MODE": "mf_cli_mode",
    "MF_VALIDATE_WAREHOUSE": "mf_validate_warehouse",
    "MF_QUERY_OUTPUT": "mf_query_output",
    "SKIP_MF_VALIDATION": "skip_mf_validation",
    "SKIP_MF_QUERY": "skip_mf_query",
    "ENABLE_DUCKDB_FALLBACK": "enable_duckdb_fallback",
    "SEMANTICFLOW_NORMALIZE_METRIC_NAMES": "normalize_metric_names",
    "SEMANTICFLOW_METRIC_NAME_PREFIX": "metric_name_prefix",
    "SEMANTICFLOW_LOG_LEVEL": "log_level",
    "SEMANTICFLOW_LOG_FORMAT": "log_format",
}


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_num(env_key: str, raw: str, kind: type):
    try:
        return kind(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{env_key}={raw!r} is not a valid {kind.__name__}"
        ) from exc


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if isinstance(data, dict) and "semanticflow" in data:
        block = data.get("semanticflow")
        if isinstance(block, dict):
            return block
    return data if isinstance(data, dict) else {}


def _read_env() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for env_key, attr in _ENV_MAP.items():
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        if attr in {"agreement_threshold", "uncertainty_threshold"}:
            values[attr] = _coerce_num(env_key, raw, float)
        elif attr in {"max_llm_repair_attempts", "llm_max_retries", "hitl_answer_budget",
                       "self_consistency_k", "experiment_repeats", "llm_max_tokens",
                       "task_timeout_sec"}:
            values[attr] = _coerce_num(env_key, raw, int)
        elif attr in {"llm_timeout_sec", "llm_repair_timeout_sec"}:
            values[attr] = _coerce_num(env_key, raw, float)
        elif attr == "run_dbt":
            values[attr] = _coerce_bool(raw)
        elif attr == "execution_consistency":
            values[attr] = _coerce_bool(raw)
        elif attr in {"llm_execution_repair", "post_execution_hitl", "slot_scoped_hitl",
                      "llm_refiner", "use_openrouter"}:
            values[attr] = _coerce_bool(raw)
        elif attr == "log_llm_raw":
            values[attr] = _coerce_bool(raw)
        elif attr == "llm_cache":
            values[attr] = _coerce_bool(raw)
        elif attr == "mf_validate_warehouse":
            values[attr] = _coerce_bool(raw)
        elif attr == "skip_mf_validation":
            values[attr] = _coerce_bool(raw)
        elif attr == "skip_mf_query":
            values[attr] = _coerce_bool(raw)
        elif attr == "enable_duckdb_fallback":
            values[attr] = _coerce_bool(raw)
        elif attr == "normalize_metric_names":
            values[attr] = _coerce_bool(raw)
        elif attr == "ensemble_providers":
            values[attr] = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            values[attr] = raw
    return values


def _apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    for key, value in overrides.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            settings.metadata[key] = value
    return settings


def load_settings(config_path: str | None = None) -> Settings:
    resolved = config_path or os.getenv("SEMANTICFLOW_CONFIG") or "config.toml"
    path = Path(resolved)
    data = _read_toml(path)
    env_overrides = _read_env()

    settings = Settings()
    settings = _apply_overrides(settings, data)
    settings = _apply_overrides(settings, env_overrides)

    if path.exists() or config_path or os.getenv("SEMANTICFLOW_CONFIG"):
        settings.config_path = str(path)
    return settings
