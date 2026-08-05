from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import date as date_type

from app.core.llm import OpenAICompatibleClient
from app.database.user_repository import DietPlanRecord, NutritionReportRecord, UserRepository


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _normalize_days(days: int) -> int:
    if isinstance(days, bool) or not isinstance(days, int):
        raise ValueError("days must be between 1 and 30")
    if days < 1 or days > 30:
        raise ValueError("days must be between 1 and 30")
    return days


def _required_date(value: str, field_name: str) -> str:
    normalized = _required_text(value, field_name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        raise ValueError(f"{field_name} must use YYYY-MM-DD")
    try:
        date_type.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc
    return normalized


def _required_payload_text(payload: dict, field_name: str, label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _optional_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    return value.strip()


class DietService:
    def __init__(self, repository: UserRepository, llm_client: OpenAICompatibleClient) -> None:
        self.repository = repository
        self.llm_client = llm_client

    async def create_plan(
        self, user_id: str, goal: str, days: int, context: str
    ) -> DietPlanRecord:
        normalized_user_id = _required_text(user_id, "user_id")
        normalized_goal = _required_text(goal, "goal")
        normalized_days = _normalize_days(days)
        normalized_context = _optional_text(context, "context")
        system_prompt = (
            "你是饮食计划智能体。基于用户目标和检索上下文生成可执行饮食计划，"
            "包含每日餐次、替换建议、营养关注点。不要编造不存在的菜谱来源。"
        )
        user_prompt = (
            f"用户：{normalized_user_id}\n"
            f"目标：{normalized_goal}\n"
            f"天数：{normalized_days}\n"
            f"检索上下文：\n{normalized_context}"
        )
        content = (
            await self.llm_client.complete_text(
                system_prompt, user_prompt, model="reasoning", temperature=0.2
            )
        ).strip()
        if not content:
            raise ValueError("diet plan content is required")
        return self.repository.create_diet_plan(
            normalized_user_id, normalized_goal, normalized_days, content
        )

    async def analyze_nutrition(self, user_id: str, date: str) -> NutritionReportRecord:
        normalized_user_id = _required_text(user_id, "user_id")
        normalized_date = _required_date(date, "date")
        checkins = self.repository.list_meal_checkins(normalized_user_id, limit=20)
        diet_plans = self.repository.list_diet_plans(normalized_user_id, limit=5)
        system_prompt = (
            "你是营养分析智能体。根据用户近期饮食计划和餐食打卡生成精细化营养看板，"
            "需要对照计划目标指出执行偏差、营养风险和下一步调整建议。"
            "输出 JSON，字段 content 为中文分析正文，metrics 为对象，包含 protein、carbs、fat、energy、risk。"
        )
        user_prompt = json.dumps(
            {
                "user_id": normalized_user_id,
                "date": normalized_date,
                "diet_plans": [asdict(plan) for plan in diet_plans],
                "checkins": [asdict(checkin) for checkin in checkins],
            },
            ensure_ascii=False,
        )
        payload = await self.llm_client.complete_json(
            system_prompt, user_prompt, model="reasoning", temperature=0.1
        )
        content = _required_payload_text(
            payload, "content", "nutrition analysis content"
        )
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ValueError("nutrition analysis metrics must be an object")
        metrics = dict(raw_metrics)
        required_metrics = {"protein", "carbs", "fat", "energy", "risk"}
        missing_metrics = sorted(required_metrics - set(metrics))
        if missing_metrics:
            raise ValueError(
                f"nutrition analysis metrics missing: {', '.join(missing_metrics)}"
            )
        for key in required_metrics:
            metric = metrics[key]
            if isinstance(metric, bool) or not isinstance(metric, str | int | float):
                raise ValueError(
                    "nutrition analysis metrics must be strings or numbers"
                )
            if isinstance(metric, int | float) and not math.isfinite(metric):
                raise ValueError("nutrition analysis metrics must be finite")
            if isinstance(metric, str):
                normalized_metric = metric.strip()
                if not normalized_metric:
                    raise ValueError(f"nutrition analysis metrics blank: {key}")
                metrics[key] = normalized_metric
        return self.repository.save_nutrition_report(
            user_id=normalized_user_id,
            date=normalized_date,
            content=content,
            metrics=metrics,
        )
