from __future__ import annotations

import json
import math
from typing import Any

import httpx


class LLMConfigurationError(RuntimeError):
    pass


def _required_secret(value: object, field_name: str) -> str:
    if value is None:
        raise LLMConfigurationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise LLMConfigurationError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise LLMConfigurationError(f"{field_name} is required")
    return normalized


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _chat_model_selector(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("chat model selector must be text")
    normalized = value.strip()
    if normalized not in {"fast", "reasoning"}:
        raise ValueError("chat model selector must be one of: fast, reasoning")
    return normalized


def _chat_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("chat temperature must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 2.0:
        raise ValueError("chat temperature must be between 0 and 2")
    return normalized


class OpenAICompatibleClient:
    """OpenAI-compatible client for Qwen, DeepSeek, and embedding services."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        fast_model: str,
        reasoning_model: str,
        embedding_api_key: str | None,
        embedding_base_url: str,
        embedding_model: str,
    ) -> None:
        self.api_key = _required_secret(api_key, "LLM_API_KEY")
        self.embedding_api_key = _required_secret(
            embedding_api_key, "EMBEDDING_API_KEY"
        )
        self.base_url = _required_text(base_url, "base_url").rstrip("/")
        self.fast_model = _required_text(fast_model, "fast_model")
        self.reasoning_model = _required_text(reasoning_model, "reasoning_model")
        self.embedding_base_url = _required_text(
            embedding_base_url, "embedding_base_url"
        ).rstrip("/")
        self.embedding_model = _required_text(embedding_model, "embedding_model")

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.1,
    ) -> str:
        normalized_system_prompt = _required_text(system_prompt, "system_prompt")
        normalized_user_prompt = _required_text(user_prompt, "user_prompt")
        normalized_model = _chat_model_selector(model)
        normalized_temperature = _chat_temperature(temperature)
        payload = {
            "model": self.fast_model if normalized_model == "fast" else self.reasoning_model,
            "messages": [
                {"role": "system", "content": normalized_system_prompt},
                {"role": "user", "content": normalized_user_prompt},
            ],
            "temperature": normalized_temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM_API request failed: {exc}") from exc
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("chat completion content must be a non-empty string")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"LLM_API response invalid: {exc}") from exc

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        text = await self.complete_text(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
        )
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return self._loads_json_object(cleaned.strip())

    @staticmethod
    def _loads_json_object(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = json.loads(OpenAICompatibleClient._extract_json_object(text))
        if not isinstance(payload, dict):
            raise ValueError("LLM JSON response must be an object")
        return payload

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError("LLM response does not contain a JSON object")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise ValueError("LLM response contains an incomplete JSON object")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("embedding inputs are required")
        normalized_texts: list[str] = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("embedding input text must be a string")
            stripped = text.strip()
            if not stripped:
                raise ValueError("embedding input text is required")
            normalized_texts.append(stripped)
        payload = {"model": self.embedding_model, "input": normalized_texts}
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.embedding_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.embedding_api_key}"},
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"EMBEDDING_API request failed: {exc}") from exc
        try:
            data = response.json()["data"]
            sorted_items = sorted(data, key=lambda item: item["index"])
            indexes = [item["index"] for item in sorted_items]
            if indexes != list(range(len(texts))):
                raise ValueError("embedding response indexes must match input texts")
            vectors = [item["embedding"] for item in sorted_items]
            if any(
                not isinstance(vector, list)
                or not vector
                or not all(self._is_embedding_number(value) for value in vector)
                for vector in vectors
            ):
                raise ValueError("embedding response must contain non-empty numeric vectors")
            return vectors
        except Exception as exc:
            raise RuntimeError(f"EMBEDDING_API response invalid: {exc}") from exc

    @staticmethod
    def _is_embedding_number(value: Any) -> bool:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
