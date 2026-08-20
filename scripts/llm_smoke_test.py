from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semanticflow.agents.semantic_mapper import SemanticMapperAgent
from semanticflow.config import load_settings
from semanticflow.llm.base import LLMError
from semanticflow.llm.router import build_provider_client


DEFAULT_PROVIDERS = ("openai", "anthropic", "gemini")


def _resolve_providers(arg: str | None) -> list[str]:
    if arg:
        return [item.strip() for item in arg.split(",") if item.strip()]
    env = os.getenv("LLM_PROVIDERS")
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    return list(DEFAULT_PROVIDERS)


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _format_exception(exc: BaseException, debug: bool) -> str:
    if debug:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    detail = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause:
        detail = f"{detail} (caused by {type(cause).__name__}: {cause})"
    return detail


def _print_preflight(settings, providers: list[str]) -> None:
    _print_header("Preflight")
    env_path = REPO_ROOT / ".env"
    print(f"Repo: {REPO_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Providers: {', '.join(providers)}")
    print(f".env present: {'yes' if env_path.exists() else 'no'}")
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'missing'}")
    print(f"ANTHROPIC_API_KEY: {'set' if os.getenv('ANTHROPIC_API_KEY') else 'missing'}")
    print(f"GEMINI_API_KEY: {'set' if os.getenv('GEMINI_API_KEY') else 'missing'}")
    base_url = os.getenv("OPENAI_BASE_URL") or "default"
    print(f"OPENAI_BASE_URL: {base_url}")
    print(
        "Models: "
        f"openai={settings.openai_model}, "
        f"anthropic={settings.anthropic_model}, "
        f"gemini={settings.gemini_model}"
    )
    print(f"LLM timeout sec: {settings.llm_timeout_sec}")
    if env_path.exists() and not any(
        os.getenv(key) for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    ):
        print("Note: .env exists but env vars are not loaded. Run `source .env`.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LLM calls and JSON repair.")
    parser.add_argument(
        "--providers",
        help="Comma-separated providers (default: openai,anthropic,gemini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use for LLM calls (default: 1.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks for errors.",
    )
    args = parser.parse_args()

    settings = load_settings()
    agent = SemanticMapperAgent(settings=settings, llm_clients={})
    providers = _resolve_providers(args.providers)
    _print_preflight(settings, providers)

    system_prompt = "You return JSON only."
    user_prompt = (
        "Return a JSON object with fields: metric_name (string), base_measures (list), "
        "group_by (list), time_granularity (string or null), filters (list), "
        "definition_notes (string), confidence (0-1), uncertainty_score (0-1), "
        "ambiguity_points (list), clarifying_questions (list), order_by (list or null), "
        "limit (int or null). Use dummy values."
    )
    repair_input = "metric_name: orders_per_day, base_measures: [order_count], group_by: [order_date]"

    for provider in providers:
        _print_header(f"Provider: {provider}")
        try:
            client = build_provider_client(provider, settings)
        except LLMError as exc:
            print(f"Skipping {provider}: {_format_exception(exc, args.debug)}")
            continue

        print(f"Model: {getattr(client, '_model', 'unknown')}")
        print(f"Client: {type(client).__name__}")
        try:
            raw = client.complete(
                system_prompt,
                user_prompt,
                temperature=args.temperature,
                json_mode=True,
            )
            print("\nRaw LLM response:")
            print(raw)
            try:
                parsed = agent._extract_json(raw)
                print("\nParsed JSON:")
                print(json.dumps(parsed, indent=2, sort_keys=True))
            except Exception as exc:
                print(f"\nParse failed: {_format_exception(exc, args.debug)}")
        except Exception as exc:
            print(f"LLM call failed: {_format_exception(exc, args.debug)}")
            continue

        try:
            repaired = agent._repair_output(client, repair_input)
            print("\nRepair input:")
            print(repair_input)
            print("\nRepair output:")
            print(repaired)
        except Exception as exc:
            print(f"Repair call failed: {_format_exception(exc, args.debug)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
