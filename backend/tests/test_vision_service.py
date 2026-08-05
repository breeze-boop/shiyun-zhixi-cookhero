import json

import httpx
import pytest

from app.core.config import Settings
from app.services.vision_service import ModelScopeVisionService


class FakeVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "识别结果如下：\n"
                            "```json\n"
                            "{\"dish_name\":\"番茄炒蛋\",\"ingredients\":[\"番茄\",\"鸡蛋\"],"
                            "\"nutrition\":{\"protein\":\"18g\"},"
                            "\"advice\":[\"少油烹饪\"],\"confidence\":0.91}\n"
                            "```"
                        )
                    }
                }
            ]
        }


class FakeAsyncClient:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(self, url: str, headers: dict, json: dict) -> FakeVisionResponse:
        return FakeVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_extracts_json_from_model_prose(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", FakeAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    result = await service.analyze_food(image_url="https://example.com/meal.jpg", image_base64=None, user_goal="少油")

    assert result.dish_name == "番茄炒蛋"
    assert result.ingredients == ["番茄", "鸡蛋"]
    assert result.nutrition == {"protein": "18g"}
    assert result.advice == ["少油烹饪"]
    assert result.confidence == 0.91


class ShouldNotCallVisionAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("blank image input should not call ModelScope")


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_blank_image_inputs_before_http(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", ShouldNotCallVisionAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(ValueError, match="image_url or image_base64 is required"):
        await service.analyze_food(image_url="   ", image_base64=None, user_goal="少油")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"image_url": 123, "image_base64": None, "user_goal": "少油"},
            "image_url must be text",
        ),
        (
            {"image_url": None, "image_base64": 123, "user_goal": "少油"},
            "image_base64 must be text",
        ),
        (
            {
                "image_url": "https://example.com/meal.jpg",
                "image_base64": None,
                "user_goal": 123,
            },
            "user_goal must be text",
        ),
    ],
)
@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_non_string_text_inputs_before_http(
    monkeypatch, kwargs, message
) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", ShouldNotCallVisionAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(ValueError, match=message):
        await service.analyze_food(**kwargs)  # type: ignore[arg-type]


class RequestErrorAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "RequestErrorAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(self, url: str, headers: dict, json: dict):
        request = httpx.Request("POST", url)
        raise httpx.RequestError("network down", request=request)


@pytest.mark.asyncio
async def test_modelscope_vision_service_wraps_http_failures_for_api_error_mapping(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", RequestErrorAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API request failed") as exc_info:
        await service.analyze_food(image_url="https://example.com/meal.jpg", image_base64=None, user_goal="少油")

    assert isinstance(exc_info.value.__cause__, httpx.RequestError)

class MalformedVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "无法识别，请重试"}}]}


class MalformedAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> MalformedVisionResponse:
        return MalformedVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_wraps_malformed_model_responses(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", MalformedAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid") as exc_info:
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


class OutOfRangeConfidenceVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "{\"dish_name\":\"番茄炒蛋\",\"ingredients\":[\"番茄\",\"鸡蛋\"],"
                            "\"nutrition\":{\"protein\":\"18g\"},"
                            "\"advice\":[\"少油烹饪\"],\"confidence\":1.7}"
                        )
                    }
                }
            ]
        }


class OutOfRangeConfidenceAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> OutOfRangeConfidenceVisionResponse:
        return OutOfRangeConfidenceVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_out_of_range_confidence(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", OutOfRangeConfidenceAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )


class BooleanConfidenceVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "{\"dish_name\":\"番茄炒蛋\",\"ingredients\":[\"番茄\",\"鸡蛋\"],"
                            "\"nutrition\":{\"protein\":\"18g\"},"
                            "\"advice\":[\"少油烹饪\"],\"confidence\":true}"
                        )
                    }
                }
            ]
        }


class BooleanConfidenceAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> BooleanConfidenceVisionResponse:
        return BooleanConfidenceVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_boolean_confidence(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", BooleanConfidenceAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )


class EmptyNutritionVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dish_name": "番茄炒蛋",
                                "ingredients": ["番茄", "鸡蛋"],
                                "nutrition": {},
                                "advice": ["少油烹饪"],
                                "confidence": 0.91,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class EmptyNutritionAsyncClient(FakeAsyncClient):
    async def post(
        self, url: str, headers: dict, json: dict
    ) -> EmptyNutritionVisionResponse:
        return EmptyNutritionVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_empty_nutrition(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", EmptyNutritionAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )


class BlankStructuredFieldsVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dish_name": "   ",
                                "ingredients": ["番茄", "   "],
                                "nutrition": {"protein": "18g"},
                                "advice": ["少油烹饪", ""],
                                "confidence": 0.91,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class BlankStructuredFieldsAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> BlankStructuredFieldsVisionResponse:
        return BlankStructuredFieldsVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_blank_structured_fields(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", BlankStructuredFieldsAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )


class NonStringListItemVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dish_name": "番茄炒蛋",
                                "ingredients": ["番茄", 123],
                                "nutrition": {"protein": "18g"},
                                "advice": ["少油烹饪", 456],
                                "confidence": 0.91,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class NonStringListItemAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> NonStringListItemVisionResponse:
        return NonStringListItemVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_non_string_list_items(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", NonStringListItemAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )


class NonFiniteNutritionVisionResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dish_name": "番茄炒蛋",
                                "ingredients": ["番茄", "鸡蛋"],
                                "nutrition": {"protein": float("nan")},
                                "advice": ["少油烹饪"],
                                "confidence": 0.91,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class NonFiniteNutritionAsyncClient(FakeAsyncClient):
    async def post(self, url: str, headers: dict, json: dict) -> NonFiniteNutritionVisionResponse:
        return NonFiniteNutritionVisionResponse()


@pytest.mark.asyncio
async def test_modelscope_vision_service_rejects_non_finite_nutrition_values(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(vision_service.httpx, "AsyncClient", NonFiniteNutritionAsyncClient)
    service = ModelScopeVisionService(
        Settings(
            llm_api_key="test",
            embedding_api_key="test",
            modelscope_api_key="vision-key",
        )
    )

    with pytest.raises(RuntimeError, match="MODELSCOPE_API response invalid"):
        await service.analyze_food(
            image_url="https://example.com/meal.jpg",
            image_base64=None,
            user_goal="少油",
        )
