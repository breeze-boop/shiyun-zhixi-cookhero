import math
import re
from datetime import date as date_type
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _strip_required_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("text field must not be blank")
    return normalized


def _strip_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _strip_required_date(value: str) -> str:
    normalized = _strip_required_text(value)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        raise ValueError("date must use YYYY-MM-DD")
    try:
        date_type.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    return normalized


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        return _strip_required_text(value)

    @field_validator("user_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional_text(value)

    @field_validator("sources", mode="before")
    @classmethod
    def strip_sources(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("source selection must be text")
            source = item.strip()
            if source:
                normalized.append(source)
        return normalized

    @field_validator("enabled_tools", mode="before")
    @classmethod
    def strip_enabled_tools(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("enabled tool name must be text")
            tool = item.strip()
            if tool:
                normalized.append(tool)
        return normalized

    user_id: str | None = None
    sources: list[Literal["recipes", "personal"]] | None = None
    enabled_tools: list[str] | None = None


class ToolOut(BaseModel):
    name: str
    description: str
    provider: str
    input_schema: dict[str, Any]


class SourceOut(BaseModel):
    title: str
    dish_name: str
    category: str
    difficulty: str
    source: str
    score: float | None = None
    data_source: str


class ChatResponse(BaseModel):
    answer: str
    thought: str
    action: str
    observation: dict[str, Any]
    rewritten_query: str
    metadata_expression: str | None
    sources: list[SourceOut]
    trace: list[str]
    context_preview: str


class PersonalDocumentRequest(BaseModel):
    user_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    category: str = "个人知识"
    difficulty: str = "普通"

    @field_validator("user_id", "title", "content", "category", "difficulty")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _strip_required_text(value)


class PersonalDocumentResponse(BaseModel):
    doc_id: str
    indexed: bool


class VisionAnalyzeRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    user_goal: str | None = None

    @field_validator("image_url", "image_base64", "user_goal")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional_text(value)

    @model_validator(mode="after")
    def require_image_input(self) -> "VisionAnalyzeRequest":
        if not self.image_url and not self.image_base64:
            raise ValueError("image_url or image_base64 is required")
        return self


class VisionAnalyzeResponse(BaseModel):
    dish_name: str
    ingredients: list[str]
    nutrition: dict[str, str | float | int]
    advice: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("dish_name")
    @classmethod
    def strip_dish_name(cls, value: str) -> str:
        return _strip_required_text(value)

    @field_validator("ingredients", "advice")
    @classmethod
    def strip_required_text_list(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value]
        if not normalized or any(not item for item in normalized):
            raise ValueError("text list field must contain non-blank items")
        return normalized

    @field_validator("nutrition")
    @classmethod
    def validate_nutrition(
        cls, value: dict[str, str | float | int]
    ) -> dict[str, str | float | int]:
        if not value:
            raise ValueError("nutrition must contain at least one item")
        normalized: dict[str, str | float | int] = {}
        for key, metric in value.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                raise ValueError("nutrition keys must not be blank")
            if isinstance(metric, bool):
                raise ValueError("nutrition values must be strings or numbers")
            if isinstance(metric, str):
                normalized_metric = metric.strip()
                if not normalized_metric:
                    raise ValueError("nutrition values must not be blank")
                normalized[normalized_key] = normalized_metric
            else:
                if not math.isfinite(metric):
                    raise ValueError("nutrition values must be finite")
                normalized[normalized_key] = metric
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("confidence must be a numeric probability")
        return value


class DietPlanRequest(BaseModel):
    user_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    days: int = Field(default=7, ge=1, le=30, strict=True)
    context_query: str | None = None

    @field_validator("user_id", "goal")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _strip_required_text(value)

    @field_validator("context_query")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional_text(value)


class DietPlanResponse(BaseModel):
    plan_id: str
    user_id: str
    goal: str
    days: int
    content: str


class MealCheckinRequest(BaseModel):
    user_id: str = Field(min_length=1)
    meal_time: str = Field(min_length=1)
    description: str = Field(min_length=1)
    image_url: str | None = None

    @field_validator("user_id", "meal_time", "description")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _strip_required_text(value)

    @field_validator("image_url", "image_base64", "user_goal")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional_text(value)

    image_base64: str | None = None
    user_goal: str | None = None
    image_analysis: dict[str, Any] = Field(default_factory=dict)


class MealCheckinResponse(BaseModel):
    checkin_id: str
    user_id: str
    meal_time: str
    description: str
    image_analysis: dict[str, Any]


class NutritionAnalysisRequest(BaseModel):
    user_id: str = Field(min_length=1)
    date: str = Field(min_length=1)

    @field_validator("user_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _strip_required_text(value)

    @field_validator("date")
    @classmethod
    def strip_required_date(cls, value: str) -> str:
        return _strip_required_date(value)


class NutritionAnalysisResponse(BaseModel):
    report_id: str
    user_id: str
    date: str
    content: str
    metrics: dict[str, str | float | int]
