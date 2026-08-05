from __future__ import annotations

import math
import re
from datetime import date as date_type
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4


@dataclass(slots=True)
class DietPlanRecord:
    plan_id: str
    user_id: str
    goal: str
    days: int
    content: str


@dataclass(slots=True)
class MealCheckinRecord:
    checkin_id: str
    user_id: str
    meal_time: str
    description: str
    image_analysis: dict[str, Any]


@dataclass(slots=True)
class NutritionReportRecord:
    report_id: str
    user_id: str
    date: str
    content: str
    metrics: dict[str, Any]


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _required_date(value: object, field_name: str) -> str:
    normalized = _required_text(value, field_name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        raise ValueError(f"{field_name} must use YYYY-MM-DD")
    try:
        date_type.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc
    return normalized


def _required_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    normalized = dict(value)
    _validate_json_value(normalized, field_name)
    return normalized


def _validate_json_value(value: object, field_name: str) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int | float):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be valid JSON")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} must be valid JSON")
            _validate_json_value(item, field_name)
        return
    raise ValueError(f"{field_name} must be valid JSON")


class UserRepository:
    def __init__(self, dsn: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("PostgreSQL DSN must be text")
        self.dsn = dsn.strip()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    @contextmanager
    def _wrap_database_errors(self) -> Iterator[None]:
        try:
            yield
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"PostgreSQL operation failed: {exc}") from exc

    def create_schema(self) -> None:
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS diet_plans (
                        plan_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        days INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_diet_plans_user ON diet_plans(user_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS meal_checkins (
                        checkin_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        meal_time TEXT NOT NULL,
                        description TEXT NOT NULL,
                        image_analysis JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_meal_checkins_user ON meal_checkins(user_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS nutrition_reports (
                        report_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        report_date TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_nutrition_reports_user
                        ON nutrition_reports(user_id, report_date);
                    """
                )

    def create_diet_plan(self, user_id: str, goal: str, days: int, content: str) -> DietPlanRecord:
        plan = DietPlanRecord(
            str(uuid4()),
            _required_text(user_id, "user_id"),
            _required_text(goal, "goal"),
            _positive_int(days, "days"),
            _required_text(content, "content"),
        )
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO diet_plans(plan_id, user_id, goal, days, content)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (plan.plan_id, plan.user_id, plan.goal, plan.days, plan.content),
                )
        return plan

    def list_diet_plans(self, user_id: str, limit: int = 5) -> list[DietPlanRecord]:
        normalized_user_id = _required_text(user_id, "user_id")
        normalized_limit = _positive_int(limit, "limit")
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT plan_id, user_id, goal, days, content
                    FROM diet_plans
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (normalized_user_id, normalized_limit),
                )
                rows = cur.fetchall()
        return [self._row_to_diet_plan(row) for row in rows]

    def create_meal_checkin(
        self, user_id: str, meal_time: str, description: str, image_analysis: dict[str, Any]
    ) -> MealCheckinRecord:
        from psycopg.types.json import Jsonb

        checkin = MealCheckinRecord(
            str(uuid4()),
            _required_text(user_id, "user_id"),
            _required_text(meal_time, "meal_time"),
            _required_text(description, "description"),
            _required_object(image_analysis, "image_analysis"),
        )
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meal_checkins(checkin_id, user_id, meal_time, description, image_analysis)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        checkin.checkin_id,
                        checkin.user_id,
                        checkin.meal_time,
                        checkin.description,
                        Jsonb(checkin.image_analysis),
                    ),
                )
        return checkin

    def list_meal_checkins(self, user_id: str, limit: int = 20) -> list[MealCheckinRecord]:
        normalized_user_id = _required_text(user_id, "user_id")
        normalized_limit = _positive_int(limit, "limit")
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT checkin_id, user_id, meal_time, description, image_analysis
                    FROM meal_checkins
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (normalized_user_id, normalized_limit),
                )
                rows = cur.fetchall()
        return [self._row_to_meal_checkin(row) for row in rows]

    @staticmethod
    def _row_to_diet_plan(row: object) -> DietPlanRecord:
        try:
            plan_id, user_id, goal, days, content = row
        except (TypeError, ValueError) as exc:
            raise ValueError("diet plan row must contain all fields") from exc
        return DietPlanRecord(
            _required_text(plan_id, "plan_id"),
            _required_text(user_id, "user_id"),
            _required_text(goal, "goal"),
            _positive_int(days, "days"),
            _required_text(content, "content"),
        )

    @staticmethod
    def _row_to_meal_checkin(row: object) -> MealCheckinRecord:
        try:
            checkin_id, user_id, meal_time, description, image_analysis = row
        except (TypeError, ValueError) as exc:
            raise ValueError("meal checkin row must contain all fields") from exc
        return MealCheckinRecord(
            _required_text(checkin_id, "checkin_id"),
            _required_text(user_id, "user_id"),
            _required_text(meal_time, "meal_time"),
            _required_text(description, "description"),
            _required_object(image_analysis, "image_analysis"),
        )

    def save_nutrition_report(
        self, user_id: str, date: str, content: str, metrics: dict[str, Any]
    ) -> NutritionReportRecord:
        from psycopg.types.json import Jsonb

        report = NutritionReportRecord(
            str(uuid4()),
            _required_text(user_id, "user_id"),
            _required_date(date, "date"),
            _required_text(content, "content"),
            _required_object(metrics, "metrics"),
        )
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO nutrition_reports(report_id, user_id, report_date, content, metrics)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (report.report_id, report.user_id, report.date, report.content, Jsonb(report.metrics)),
                )
        return report
