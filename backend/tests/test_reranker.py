import pytest

from app.rag.document import Document
from app.rag.rerankers import siliconflow_reranker as reranker_module
from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker


class FakeRerankResponse:
    results = [
        {"index": 0, "relevance_score": 0.4},
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.01},
    ]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"results": self.results}


class FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs) -> FakeRerankResponse:
        return FakeRerankResponse()


class FailingRerankAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "FailingRerankAsyncClient":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, *args, **kwargs):
        raise AssertionError("invalid rerank inputs must not call SiliconFlow API")


@pytest.mark.parametrize(
    ("query", "documents", "message"),
    [
        (123, [Document("first", {"source": "first"})], "rerank query must be text"),
        ("   ", [Document("first", {"source": "first"})], "rerank query is required"),
        ("query", ["first"], "rerank documents must be Document instances"),
        ("query", [Document("   ", {"source": "blank"})], "rerank document text is required"),
    ],
)
@pytest.mark.asyncio
async def test_reranker_rejects_invalid_inputs_before_http_request(
    monkeypatch, query, documents, message
) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FailingRerankAsyncClient)
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )

    with pytest.raises(ValueError, match=message):
        await reranker.rerank(query, documents)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_reranker_treats_blank_api_key_as_disabled(monkeypatch) -> None:
    created_clients = []

    class ShouldNotCallAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            created_clients.append((args, kwargs))

    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", ShouldNotCallAsyncClient)
    reranker = SiliconFlowReranker(
        api_key="   ",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [Document("first", {"source": "first"})]

    ranked = await reranker.rerank("query", documents)

    assert ranked == documents
    assert created_clients == []


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (123, "SiliconFlow rerank model must be text"),
        ("   ", "SiliconFlow rerank model is required"),
    ],
)
def test_reranker_rejects_invalid_model_before_http_request(model, message) -> None:
    with pytest.raises(ValueError, match=message):
        SiliconFlowReranker(
            api_key="test-key",
            model=model,
            enabled=True,
            min_score=0.05,
        )


@pytest.mark.parametrize("min_score", [True, -0.01, 1.01, float("nan"), "0.05"])
def test_reranker_rejects_invalid_min_score_before_http_request(min_score) -> None:
    with pytest.raises(ValueError, match="SiliconFlow rerank min_score must be between 0 and 1"):
        SiliconFlowReranker(
            api_key="test-key",
            model="BAAI/bge-reranker-v2-m3",
            enabled=True,
            min_score=min_score,
        )


@pytest.mark.asyncio
async def test_reranker_sorts_documents_by_relevance_score(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeAsyncClient)
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [
        Document("first", {"source": "first"}),
        Document("second", {"source": "second"}),
        Document("third", {"source": "third"}),
    ]

    ranked = await reranker.rerank("query", documents)

    assert [doc.metadata["source"] for doc in ranked] == ["second", "first"]
    assert [doc.metadata["rerank_score"] for doc in ranked] == [0.9, 0.4]


@pytest.mark.asyncio
async def test_reranker_returns_empty_when_successful_scores_are_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        FakeRerankResponse,
        "results",
        [
            {"index": 0, "relevance_score": 0.01},
            {"index": 1, "relevance_score": 0.02},
        ],
    )
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [
        Document("first", {"source": "first"}),
        Document("second", {"source": "second"}),
    ]

    ranked = await reranker.rerank("query", documents)

    assert ranked == []


@pytest.mark.asyncio
async def test_reranker_degrades_when_response_repeats_document_indexes(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        FakeRerankResponse,
        "results",
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ],
    )
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [
        Document("first", {"source": "first"}),
        Document("second", {"source": "second"}),
    ]

    ranked = await reranker.rerank("query", documents)

    assert ranked == documents


@pytest.mark.asyncio
async def test_reranker_degrades_to_original_order_when_response_is_malformed(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        FakeRerankResponse,
        "results",
        [{"relevance_score": 0.9}],
    )
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [
        Document("first", {"source": "first"}),
        Document("second", {"source": "second"}),
    ]

    ranked = await reranker.rerank("query", documents)

    assert ranked == documents


@pytest.mark.asyncio
async def test_reranker_degrades_when_response_score_is_not_finite(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        FakeRerankResponse,
        "results",
        [{"index": 0, "relevance_score": "NaN"}],
    )
    reranker = SiliconFlowReranker(
        api_key="test-key",
        model="BAAI/bge-reranker-v2-m3",
        enabled=True,
        min_score=0.05,
    )
    documents = [
        Document("first", {"source": "first"}),
        Document("second", {"source": "second"}),
    ]

    ranked = await reranker.rerank("query", documents)

    assert ranked == documents
