from __future__ import annotations

import httpx

from app.core.config import Settings
from app.core.llm import OpenAICompatibleClient
from app.schemas import VisionAnalyzeResponse


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    return normalized or None


class ModelScopeVisionService:
    def __init__(self, settings: Settings) -> None:
        if not settings.modelscope_api_key:
            raise RuntimeError("MODELSCOPE_API_KEY is required for food image analysis")
        self.api_key = settings.modelscope_api_key
        self.base_url = settings.modelscope_base_url.rstrip("/")
        self.model = settings.vision_model

    async def analyze_food(
        self, *, image_url: str | None, image_base64: str | None, user_goal: str | None
    ) -> VisionAnalyzeResponse:
        normalized_url = _optional_text(image_url, "image_url")
        normalized_base64 = _optional_text(image_base64, "image_base64")
        normalized_user_goal = _optional_text(user_goal, "user_goal")
        if not normalized_url and not normalized_base64:
            raise ValueError("image_url or image_base64 is required")
        image_payload = (
            {"type": "image_url", "image_url": {"url": normalized_url}}
            if normalized_url
            else {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{normalized_base64}"}}
        )
        prompt = (
            "识别图片中的食物或菜品，估算主要食材和营养信息。"
            "只返回 JSON，字段为 dish_name、ingredients、nutrition、advice、confidence。"
            f"用户目标：{normalized_user_goal or '未提供'}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_payload,
                    ],
                }
            ],
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"MODELSCOPE_API request failed: {exc}") from exc
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
            data = OpenAICompatibleClient._loads_json_object(text)
            return VisionAnalyzeResponse(**data)
        except Exception as exc:
            raise RuntimeError(f"MODELSCOPE_API response invalid: {exc}") from exc
