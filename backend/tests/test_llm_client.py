import httpx
import pytest

from app.core import llm as llm_module
from app.core.llm import LLMConfigurationError, OpenAICompatibleClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "可以，调度结果如下：\n"
                                "{\"thought\":\"需要检索菜谱\","
                                "\"action\":\"knowledge_base_search\","
                                "\"action_input\":{\"query\":\"番茄炒蛋怎么做\"}}\n"
                                "请按该 JSON 执行。"
                            )
                        }
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": "   "}, "LLM_API_KEY is required"),
        ({"api_key": 123}, "LLM_API_KEY must be text"),
        ({"embedding_api_key": "   "}, "EMBEDDING_API_KEY is required"),
        ({"embedding_api_key": 123}, "EMBEDDING_API_KEY must be text"),
    ],
)
def test_openai_compatible_client_rejects_invalid_api_keys(overrides, message) -> None:
    kwargs = {
        "api_key": "test",
        "base_url": "https://api.example.test/v1",
        "fast_model": "fast",
        "reasoning_model": "reasoning",
        "embedding_api_key": "test",
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding",
    }
    kwargs.update(overrides)

    with pytest.raises(LLMConfigurationError, match=message):
        OpenAICompatibleClient(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "   "}, "base_url is required"),
        ({"base_url": 123}, "base_url must be text"),
        ({"fast_model": "   "}, "fast_model is required"),
        ({"reasoning_model": None}, "reasoning_model must be text"),
        ({"embedding_base_url": "   "}, "embedding_base_url is required"),
        ({"embedding_model": 123}, "embedding_model must be text"),
    ],
)
def test_openai_compatible_client_rejects_invalid_runtime_text(overrides, message) -> None:
    kwargs = {
        "api_key": "test",
        "base_url": "https://api.example.test/v1",
        "fast_model": "fast",
        "reasoning_model": "reasoning",
        "embedding_api_key": "test",
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embedding",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        OpenAICompatibleClient(**kwargs)


@pytest.mark.asyncio
async def test_complete_json_extracts_object_from_model_explanation(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    payload = await client.complete_json("system", "user")

    assert payload == {
        "thought": "需要检索菜谱",
        "action": "knowledge_base_search",
        "action_input": {"query": "番茄炒蛋怎么做"},
    }


class RequestErrorAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "RequestErrorAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs):
        request = httpx.Request("POST", str(args[0]))
        raise httpx.RequestError("network down", request=request)


class MalformedChatAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"choices": [{"message": {}}]})


class MalformedEmbeddingAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"data": [{"index": 0}]})


class NullContentChatAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"choices": [{"message": {"content": None}}]})


class InvalidEmbeddingVectorAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"data": [{"index": 0, "embedding": None}]})


class MissingEmbeddingItemAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})


class BooleanEmbeddingVectorAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"data": [{"index": 0, "embedding": [True, False, True]}]})


class NonFiniteEmbeddingVectorAsyncClient(FakeAsyncClient):
    async def post(self, *args, **kwargs) -> FakeResponse:
        return FakeResponse({"data": [{"index": 0, "embedding": [0.1, float("nan"), 0.3]}]})


@pytest.mark.asyncio
async def test_complete_text_wraps_chat_http_failures_for_api_error_mapping(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", RequestErrorAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="LLM_API request failed") as exc_info:
        await client.complete_text("system", "user")

    assert isinstance(exc_info.value.__cause__, httpx.RequestError)


class FailingRequestAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "FailingRequestAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs):
        raise AssertionError("embedding request should not be sent for invalid inputs")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"system_prompt": 123, "user_prompt": "user"},
            "system_prompt must be text",
        ),
        (
            {"system_prompt": "system", "user_prompt": None},
            "user_prompt must be text",
        ),
        (
            {"system_prompt": "system", "user_prompt": "user", "model": "slow"},
            "chat model selector must be one of",
        ),
        (
            {"system_prompt": "system", "user_prompt": "user", "temperature": True},
            "chat temperature must be a finite number",
        ),
        (
            {"system_prompt": "system", "user_prompt": "user", "temperature": float("nan")},
            "chat temperature must be between 0 and 2",
        ),
        (
            {"system_prompt": "system", "user_prompt": "user", "temperature": -0.1},
            "chat temperature must be between 0 and 2",
        ),
        (
            {"system_prompt": "system", "user_prompt": "user", "temperature": 2.1},
            "chat temperature must be between 0 and 2",
        ),
    ],
)
@pytest.mark.asyncio
async def test_complete_text_rejects_invalid_inputs_before_http_request(
    monkeypatch, kwargs, message
) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FailingRequestAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(ValueError, match=message):
        await client.complete_text(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embed_documents_rejects_empty_inputs_before_http_request(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FailingRequestAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(ValueError, match="embedding inputs are required"):
        await client.embed_documents([])

    with pytest.raises(ValueError, match="embedding input text is required"):
        await client.embed_documents(["番茄炒蛋", "   "])


@pytest.mark.asyncio
async def test_embedding_inputs_reject_non_string_values_before_http_request(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FailingRequestAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(ValueError, match="embedding input text must be a string"):
        await client.embed_documents(["番茄炒蛋", 123])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="embedding input text must be a string"):
        await client.embed_query(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embed_documents_wraps_embedding_http_failures_for_api_error_mapping(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", RequestErrorAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API request failed") as exc_info:
        await client.embed_documents(["番茄炒蛋"])

    assert isinstance(exc_info.value.__cause__, httpx.RequestError)


def test_loads_json_object_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        OpenAICompatibleClient._loads_json_object("[1, 2, 3]")


@pytest.mark.asyncio
async def test_complete_text_wraps_malformed_chat_responses_for_api_error_mapping(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", MalformedChatAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="LLM_API response invalid") as exc_info:
        await client.complete_text("system", "user")

    assert isinstance(exc_info.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_embed_documents_wraps_malformed_embedding_responses_for_api_error_mapping(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", MalformedEmbeddingAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API response invalid") as exc_info:
        await client.embed_documents(["番茄炒蛋"])

    assert isinstance(exc_info.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_complete_text_rejects_null_chat_content(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", NullContentChatAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="LLM_API response invalid") as exc_info:
        await client.complete_text("system", "user")

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_embed_documents_rejects_non_vector_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", InvalidEmbeddingVectorAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API response invalid") as exc_info:
        await client.embed_documents(["番茄炒蛋"])

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_embed_documents_rejects_missing_embedding_items(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", MissingEmbeddingItemAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API response invalid") as exc_info:
        await client.embed_documents(["番茄炒蛋", "红烧肉"])

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_embed_documents_rejects_boolean_embedding_values(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", BooleanEmbeddingVectorAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API response invalid") as exc_info:
        await client.embed_documents(["番茄炒蛋"])

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_embed_documents_rejects_non_finite_embedding_values(monkeypatch) -> None:
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", NonFiniteEmbeddingVectorAsyncClient)
    client = OpenAICompatibleClient(
        api_key="test",
        base_url="https://api.example.test/v1",
        fast_model="fast",
        reasoning_model="reasoning",
        embedding_api_key="test",
        embedding_base_url="https://api.example.test/v1",
        embedding_model="embedding",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API response invalid") as exc_info:
        await client.embed_documents(["番茄炒蛋"])

    assert isinstance(exc_info.value.__cause__, ValueError)
