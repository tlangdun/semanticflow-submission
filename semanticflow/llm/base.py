from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel


class LLMError(RuntimeError):
    pass


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseLLMClient:
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        temperature: float | None = None,
    ) -> SchemaT:
        raw = self.complete(system_prompt, user_prompt, temperature=temperature, json_mode=True)
        payload = self._extract_json(raw)
        return schema.model_validate(payload)

    def _extract_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise LLMError("LLM did not return JSON")
            return json.loads(match.group(0))
