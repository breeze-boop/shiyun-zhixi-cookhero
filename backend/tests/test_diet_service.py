import pytest

from app.services.diet_service import DietService
from tests.fakes import FakeLLMClient, FakeUserRepository


@pytest.mark.asyncio
async def test_diet_service_generates_plan_records_checkin_and_analyzes_nutrition() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=FakeLLMClient())

    plan = await service.create_plan(user_id="u1", goal="减脂高蛋白", days=3, context="番茄炒蛋")
    checkin = repository.create_meal_checkin(
        user_id="u1", meal_time="dinner", description="番茄炒蛋和米饭", image_analysis={"protein": "24g"}
    )
    report = await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert plan.user_id == "u1"
    assert "减脂高蛋白" in plan.goal
    assert checkin.description == "番茄炒蛋和米饭"
    assert report.user_id == "u1"
    assert repository.list_meal_checkins("u1", limit=5)[0].meal_time == "dinner"


class CapturingNutritionLLM(FakeLLMClient):
    def __init__(self) -> None:
        self.last_user_prompt = ""

    async def complete_json(self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.0) -> dict:
        self.last_user_prompt = user_prompt
        return await super().complete_json(system_prompt, user_prompt, model=model, temperature=temperature)


class CapturingDietPlanLLM(FakeLLMClient):
    def __init__(self) -> None:
        self.last_user_prompt = ""

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.1,
    ) -> str:
        self.last_user_prompt = user_prompt
        return await super().complete_text(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_diet_plan_service_normalizes_inputs_before_prompt_and_saving() -> None:
    repository = FakeUserRepository()
    llm_client = CapturingDietPlanLLM()
    service = DietService(repository=repository, llm_client=llm_client)

    plan = await service.create_plan(
        user_id=" u1 ", goal=" 减脂高蛋白 ", days=3, context=" 番茄炒蛋 "
    )

    assert plan.user_id == "u1"
    assert plan.goal == "减脂高蛋白"
    assert "用户：u1" in llm_client.last_user_prompt
    assert "目标：减脂高蛋白" in llm_client.last_user_prompt
    assert "番茄炒蛋" in llm_client.last_user_prompt


class BlankDietPlanLLM(FakeLLMClient):
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.1,
    ) -> str:
        if "饮食计划智能体" in system_prompt:
            return "   "
        return await super().complete_text(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_diet_plan_service_rejects_blank_llm_plan_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=BlankDietPlanLLM())

    with pytest.raises(ValueError, match="diet plan content is required"):
        await service.create_plan(user_id="u1", goal="减脂", days=3, context="番茄炒蛋")

    assert repository.plans == []


class NoCallDietPlanLLM(FakeLLMClient):
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.1,
    ) -> str:
        raise AssertionError("invalid diet plan inputs must not call LLM_API")


@pytest.mark.asyncio
async def test_diet_plan_service_rejects_non_string_text_before_llm_call() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=NoCallDietPlanLLM())

    with pytest.raises(ValueError, match="user_id must be text"):
        await service.create_plan(user_id=None, goal="减脂", days=3, context="")
    with pytest.raises(ValueError, match="context must be text"):
        await service.create_plan(user_id="u1", goal="减脂", days=3, context=123)  # type: ignore[arg-type]

    assert repository.plans == []


@pytest.mark.asyncio
async def test_diet_plan_service_rejects_invalid_core_fields_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=FakeLLMClient())

    with pytest.raises(ValueError, match="user_id is required"):
        await service.create_plan(user_id="   ", goal="减脂", days=3, context="")
    with pytest.raises(ValueError, match="goal is required"):
        await service.create_plan(user_id="u1", goal="   ", days=3, context="")
    for invalid_days in (0, True, 1.5, "3"):
        with pytest.raises(ValueError, match="days must be between 1 and 30"):
            await service.create_plan(user_id="u1", goal="减脂", days=invalid_days, context="")  # type: ignore[arg-type]

    assert repository.plans == []


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_invalid_core_fields_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=FakeLLMClient())

    with pytest.raises(ValueError, match="user_id is required"):
        await service.analyze_nutrition(user_id="   ", date="2026-07-30")
    with pytest.raises(ValueError, match="date is required"):
        await service.analyze_nutrition(user_id="u1", date="   ")
    for invalid_date in ("2026/07/30", "2026-7-30", "2026-02-30"):
        with pytest.raises(ValueError, match="date must use YYYY-MM-DD"):
            await service.analyze_nutrition(user_id="u1", date=invalid_date)

    assert repository.reports == []


@pytest.mark.asyncio
async def test_nutrition_analysis_normalizes_user_and_date_before_prompt_and_saving() -> None:
    repository = FakeUserRepository()
    llm_client = CapturingNutritionLLM()
    service = DietService(repository=repository, llm_client=llm_client)
    await service.create_plan(user_id="u1", goal="减脂高蛋白", days=3, context="番茄炒蛋")
    repository.create_meal_checkin(
        user_id="u1", meal_time="dinner", description="番茄炒蛋和米饭", image_analysis={"protein": "24g"}
    )

    report = await service.analyze_nutrition(user_id=" u1 ", date=" 2026-07-30 ")

    assert report.user_id == "u1"
    assert report.date == "2026-07-30"
    assert "番茄炒蛋和米饭" in llm_client.last_user_prompt
    assert '"user_id": "u1"' in llm_client.last_user_prompt
    assert '"date": "2026-07-30"' in llm_client.last_user_prompt


@pytest.mark.asyncio
async def test_nutrition_analysis_includes_recent_diet_plans_in_llm_prompt() -> None:
    repository = FakeUserRepository()
    llm_client = CapturingNutritionLLM()
    service = DietService(repository=repository, llm_client=llm_client)
    await service.create_plan(user_id="u1", goal="减脂高蛋白", days=3, context="番茄炒蛋")
    repository.create_meal_checkin(
        user_id="u1", meal_time="dinner", description="番茄炒蛋和米饭", image_analysis={"protein": "24g"}
    )

    await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert "diet_plans" in llm_client.last_user_prompt
    assert "减脂高蛋白" in llm_client.last_user_prompt


class MalformedNutritionLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {"content": "", "metrics": {"protein": "24g"}}
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_incomplete_llm_report_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=MalformedNutritionLLM())

    with pytest.raises(ValueError, match="nutrition analysis content is required"):
        await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert repository.reports == []


class NonStringContentNutritionLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {
                "content": 123,
                "metrics": {
                    "protein": "24g",
                    "carbs": "38g",
                    "fat": "16g",
                    "energy": "620kcal",
                    "risk": "low",
                },
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_non_string_llm_content_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=NonStringContentNutritionLLM())

    with pytest.raises(ValueError, match="nutrition analysis content must be text"):
        await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert repository.reports == []


class BooleanMetricsNutritionLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {
                "content": "今日整体摄入偏油，需要补充蔬菜。",
                "metrics": {
                    "protein": "24g",
                    "carbs": "38g",
                    "fat": True,
                    "energy": "620kcal",
                    "risk": "low",
                },
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_boolean_metric_values_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=BooleanMetricsNutritionLLM())

    with pytest.raises(
        ValueError, match="nutrition analysis metrics must be strings or numbers"
    ):
        await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert repository.reports == []


class NonFiniteMetricsNutritionLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {
                "content": "今日整体摄入偏油，需要补充蔬菜。",
                "metrics": {
                    "protein": float("nan"),
                    "carbs": "38g",
                    "fat": "16g",
                    "energy": "620kcal",
                    "risk": "low",
                },
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_non_finite_metric_values_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=NonFiniteMetricsNutritionLLM())

    with pytest.raises(ValueError, match="nutrition analysis metrics must be finite"):
        await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert repository.reports == []


class BlankMetricsNutritionLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {
                "content": "今日整体摄入偏油，需要补充蔬菜。",
                "metrics": {
                    "protein": "",
                    "carbs": "   ",
                    "fat": "",
                    "energy": "",
                    "risk": "",
                },
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_nutrition_analysis_rejects_blank_metric_values_before_saving() -> None:
    repository = FakeUserRepository()
    service = DietService(repository=repository, llm_client=BlankMetricsNutritionLLM())

    with pytest.raises(ValueError, match="nutrition analysis metrics blank"):
        await service.analyze_nutrition(user_id="u1", date="2026-07-30")

    assert repository.reports == []


class DatabaseUnavailable(Exception):
    pass


class UserRowsCursor:
    def __init__(self, rows) -> None:
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None) -> None:
        return None

    def fetchall(self):
        return self.rows


class UserRowsConnection:
    def __init__(self, cursor: UserRowsCursor) -> None:
        self.cursor_obj = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


@pytest.mark.parametrize("dsn", ["", "   ", 123])
def test_user_repository_rejects_invalid_dsn(dsn) -> None:
    from app.database.user_repository import UserRepository

    with pytest.raises(ValueError, match="PostgreSQL DSN must be text"):
        UserRepository(dsn)

@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("plan-1", "u1", "   ", 3, "计划内容"), "goal is required"),
        (("plan-1", "u1", "减脂", 0, "计划内容"), "days must be positive"),
        (("plan-1", "u1", "减脂", True, "计划内容"), "days must be positive"),
        (("plan-1", "   ", "减脂", 3, "计划内容"), "user_id is required"),
    ],
)
def test_user_repository_rejects_malformed_diet_plan_rows_from_database(monkeypatch, row, message) -> None:
    from app.database.user_repository import UserRepository

    cursor = UserRowsCursor([row])
    repository = UserRepository("postgresql://example")
    monkeypatch.setattr(repository, "_connect", lambda: UserRowsConnection(cursor))

    with pytest.raises(ValueError, match=message):
        repository.list_diet_plans("u1")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("checkin-1", "u1", "   ", "番茄炒蛋", {}), "meal_time is required"),
        (("checkin-1", "u1", "dinner", "   ", {}), "description is required"),
        (("checkin-1", "u1", "dinner", "番茄炒蛋", []), "image_analysis must be an object"),
        (("checkin-1", "u1", "dinner", "番茄炒蛋", {1: "protein"}), "image_analysis must be valid JSON"),
    ],
)
def test_user_repository_rejects_malformed_meal_checkin_rows_from_database(monkeypatch, row, message) -> None:
    from app.database.user_repository import UserRepository

    cursor = UserRowsCursor([row])
    repository = UserRepository("postgresql://example")
    monkeypatch.setattr(repository, "_connect", lambda: UserRowsConnection(cursor))

    with pytest.raises(ValueError, match=message):
        repository.list_meal_checkins("u1")


def test_user_repository_rejects_blank_required_fields_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid user records")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_diet_plan("   ", "减脂", 3, "content"),
        lambda: repository.create_diet_plan("u1", "   ", 3, "content"),
        lambda: repository.create_diet_plan("u1", "减脂", 3, "   "),
        lambda: repository.list_diet_plans("   "),
        lambda: repository.create_meal_checkin("   ", "dinner", "番茄炒蛋", {}),
        lambda: repository.create_meal_checkin("u1", "   ", "番茄炒蛋", {}),
        lambda: repository.create_meal_checkin("u1", "dinner", "   ", {}),
        lambda: repository.list_meal_checkins("   "),
        lambda: repository.save_nutrition_report("   ", "2026-07-30", "content", {"risk": "low"}),
        lambda: repository.save_nutrition_report("u1", "   ", "content", {"risk": "low"}),
        lambda: repository.save_nutrition_report("u1", "2026-07-30", "   ", {"risk": "low"}),
    ]

    for operation in operations:
        with pytest.raises(ValueError, match="is required"):
            operation()


def test_user_repository_rejects_non_string_text_fields_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid text fields")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    with pytest.raises(ValueError, match="user_id must be text"):
        repository.create_diet_plan(None, "减脂", 3, "content")


def test_user_repository_rejects_invalid_numeric_fields_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid user record numbers")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_diet_plan("u1", "减脂", 0, "content"),
        lambda: repository.create_diet_plan("u1", "减脂", True, "content"),
        lambda: repository.list_diet_plans("u1", limit=0),
        lambda: repository.list_diet_plans("u1", limit=True),
        lambda: repository.list_meal_checkins("u1", limit=0),
        lambda: repository.list_meal_checkins("u1", limit=True),
    ]

    for operation in operations:
        with pytest.raises(ValueError, match="must be positive"):
            operation()


def test_user_repository_rejects_invalid_nutrition_report_dates_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid nutrition report dates")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    for invalid_date in ("2026/07/30", "2026-7-30", "2026-02-30"):
        with pytest.raises(ValueError, match="date must use YYYY-MM-DD"):
            repository.save_nutrition_report("u1", invalid_date, "content", {"risk": "low"})


def test_user_repository_rejects_non_object_json_fields_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid JSON fields")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_meal_checkin("u1", "dinner", "番茄炒蛋", ["not", "object"]),
        lambda: repository.save_nutrition_report("u1", "2026-07-30", "content", ["not", "object"]),
    ]

    for operation in operations:
        with pytest.raises(ValueError, match="must be an object"):
            operation()


def test_user_repository_rejects_invalid_json_payload_values_before_database_calls(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("database should not be called for invalid JSON payloads")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_meal_checkin("u1", "dinner", "番茄炒蛋", {1: "protein"}),
        lambda: repository.create_meal_checkin("u1", "dinner", "番茄炒蛋", {"confidence": float("nan")}),
        lambda: repository.save_nutrition_report(
            "u1", "2026-07-30", "content", {"nested": {"energy": float("inf")}}
        ),
        lambda: repository.save_nutrition_report("u1", "2026-07-30", "content", {"raw": object()}),
    ]

    for operation in operations:
        with pytest.raises(ValueError, match="must be valid JSON"):
            operation()


def test_user_repository_wraps_database_failures_for_api_error_mapping(monkeypatch) -> None:
    from app.database.user_repository import UserRepository

    repository = UserRepository("postgresql://example")

    def fail_connect():
        raise DatabaseUnavailable("postgres down")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_diet_plan("u1", "减脂", 3, "content"),
        lambda: repository.list_diet_plans("u1"),
        lambda: repository.create_meal_checkin("u1", "dinner", "番茄炒蛋", {}),
        lambda: repository.list_meal_checkins("u1"),
        lambda: repository.save_nutrition_report("u1", "2026-07-30", "content", {}),
    ]

    for operation in operations:
        with pytest.raises(RuntimeError, match="PostgreSQL operation failed") as exc_info:
            operation()
        assert isinstance(exc_info.value.__cause__, DatabaseUnavailable)
