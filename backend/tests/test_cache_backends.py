import asyncio
import pickle
import sys
from types import SimpleNamespace

import pytest

from app.rag.cache.backends import MilvusVectorCache, RedisKeywordCache
from app.rag.document import Document
from tests.fakes import FakeLLMClient


@pytest.mark.parametrize("milvus_uri", ["", "   ", 123])
def test_milvus_vector_cache_rejects_invalid_milvus_uri(milvus_uri) -> None:
    with pytest.raises(ValueError, match="Milvus cache URI must be text"):
        MilvusVectorCache(
            milvus_uri=milvus_uri,
            collection_name="cache",
            embedding_client=FakeLLMClient(),
            embedding_dim=3,
        )


@pytest.mark.parametrize("collection_name", ["", "   ", 123])
def test_milvus_vector_cache_rejects_invalid_collection_name(collection_name) -> None:
    with pytest.raises(ValueError, match="Milvus cache collection name must be text"):
        MilvusVectorCache(
            milvus_uri="http://localhost:19530",
            collection_name=collection_name,
            embedding_client=FakeLLMClient(),
            embedding_dim=3,
        )


@pytest.mark.parametrize("embedding_dim", [0, -1, True, 1.5, "3"])
def test_milvus_vector_cache_rejects_invalid_embedding_dim(embedding_dim) -> None:
    with pytest.raises(
        ValueError, match="Milvus cache embedding_dim must be a positive integer"
    ):
        MilvusVectorCache(
            milvus_uri="http://localhost:19530",
            collection_name="cache",
            embedding_client=FakeLLMClient(),
            embedding_dim=embedding_dim,
        )


def test_milvus_vector_cache_payload_round_trips_compressed_documents() -> None:
    documents = [
        Document(
            page_content="番茄炒蛋" * 2000,
            metadata={"parent_id": "doc-1", "dish_name": "番茄炒蛋", "score": 0.92},
        )
    ]
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )

    payload = cache.encode_payload(documents)
    restored = cache.decode_payload(payload)

    assert restored == documents
    assert len(payload) < len(pickle.dumps(documents).hex())


@pytest.mark.parametrize("redis_url", ["", "   ", 123])
def test_redis_keyword_cache_rejects_invalid_redis_url_before_client_creation(
    redis_url,
    monkeypatch,
) -> None:
    calls = []

    class FakeRedisModule:
        @staticmethod
        def from_url(value: str):
            calls.append(value)
            raise AssertionError("invalid Redis URL should not create Redis client")

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedisModule))

    with pytest.raises(ValueError, match="Redis cache URL must be text"):
        RedisKeywordCache(redis_url)

    assert calls == []


def test_redis_keyword_cache_rejects_non_positive_ttl_before_write(monkeypatch) -> None:
    calls = []

    class FakeRedisClient:
        def setex(self, *args):
            calls.append(args)

    class FakeRedisModule:
        @staticmethod
        def from_url(redis_url: str) -> FakeRedisClient:
            return FakeRedisClient()

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedisModule))
    cache = RedisKeywordCache("redis://localhost:6379/0")

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "global",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=0,
        )

    with pytest.raises(ValueError, match="retrieval cache ttl must be positive"):
        asyncio.run(run_set())

    assert calls == []


def test_redis_keyword_cache_rejects_blank_scope_before_write(monkeypatch) -> None:
    calls = []

    class FakeRedisClient:
        def setex(self, *args):
            calls.append(args)

    class FakeRedisModule:
        @staticmethod
        def from_url(redis_url: str) -> FakeRedisClient:
            return FakeRedisClient()

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedisModule))
    cache = RedisKeywordCache("redis://localhost:6379/0")

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "   ",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(ValueError, match="retrieval cache field scope is required"):
        asyncio.run(run_set())

    assert calls == []


class CapturingMilvusClient:
    def __init__(self) -> None:
        self.filter = ""

    def search(self, **kwargs):
        self.filter = kwargs["filter"]
        return []


async def _run_cache_get(cache: MilvusVectorCache) -> None:
    await cache.get("recipes", 'user"\\id', "番茄炒蛋", threshold=0.92)


@pytest.mark.parametrize(
    "threshold", [0.0, -0.1, 1.01, True, float("nan"), float("inf"), "0.92"]
)
def test_milvus_vector_cache_rejects_invalid_threshold_before_external_calls(
    threshold,
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

        def search(self, **kwargs):
            raise AssertionError("invalid cache threshold should not search Milvus")

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_get() -> None:
        await cache.get("recipes", "global", "番茄炒蛋", threshold=threshold)

    with pytest.raises(
        ValueError, match="Milvus cache threshold must be between 0 and 1"
    ):
        asyncio.run(run_get())

    assert created_clients == []


def test_milvus_vector_cache_rejects_blank_scope_before_lookup() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_get() -> None:
        await cache.get("recipes", "   ", "番茄炒蛋", threshold=0.92)

    with pytest.raises(ValueError, match="retrieval cache field scope is required"):
        asyncio.run(run_get())


def test_milvus_vector_cache_escapes_scope_in_filter() -> None:
    client = CapturingMilvusClient()
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = client

    asyncio.run(_run_cache_get(cache))

    assert 'scope == "user\\"\\\\id"' in client.filter


class WrongDimensionEmbeddingClient(FakeLLMClient):
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class RejectingMilvusClient:
    def insert(self, **kwargs):
        raise AssertionError("insert should not run after invalid cache embedding")


def test_milvus_vector_cache_rejects_query_vectors_with_wrong_dimension() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=WrongDimensionEmbeddingClient(),
        embedding_dim=3,
    )
    cache._client = RejectingMilvusClient()

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "global",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(RuntimeError, match="embedding vector dimension mismatch"):
        asyncio.run(run_set())


class CapturingEmbeddingClient(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


class CapturingInsertMilvusClient:
    def __init__(self) -> None:
        self.rows = []

    def insert(self, **kwargs):
        self.rows.extend(kwargs["data"])


def test_milvus_vector_cache_embeds_normalized_query_on_write() -> None:
    embedding_client = CapturingEmbeddingClient()
    milvus_client = CapturingInsertMilvusClient()
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=embedding_client,
        embedding_dim=3,
    )
    cache._client = milvus_client

    asyncio.run(
        cache.set(
            "recipes",
            "global",
            "  番茄炒蛋  ",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )
    )

    assert embedding_client.queries == ["番茄炒蛋"]
    assert milvus_client.rows[0]["query"] == "番茄炒蛋"


class NoCallEmbeddingClient:
    async def embed_query(self, text: str) -> list[float]:
        raise AssertionError("invalid cache rows should not call embedding API")


def test_milvus_vector_cache_rejects_non_positive_ttl_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "global",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=0,
        )

    with pytest.raises(ValueError, match="retrieval cache ttl must be positive"):
        asyncio.run(run_set())

    assert created_clients == []


def test_milvus_vector_cache_rejects_overlong_source_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            "s" * 65,
            "global",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(
        ValueError, match="Milvus cache field source exceeds max length 64"
    ):
        asyncio.run(run_set())

    assert created_clients == []


def test_milvus_vector_cache_rejects_blank_scope_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "   ",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(ValueError, match="retrieval cache field scope is required"):
        asyncio.run(run_set())

    assert created_clients == []


def test_milvus_vector_cache_rejects_non_text_identity_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            123,
            "global",
            "番茄炒蛋",
            [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(ValueError, match="retrieval cache field source must be text"):
        asyncio.run(run_set())

    assert created_clients == []


def test_milvus_vector_cache_rejects_malformed_documents_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "global",
            "番茄炒蛋",
            [Document(page_content="   ", metadata={"parent_id": "doc-1"})],
            ttl=3600,
        )

    with pytest.raises(ValueError, match="retrieval cache document content must be text"):
        asyncio.run(run_set())

    assert created_clients == []


def test_milvus_vector_cache_rejects_non_text_document_identity_metadata_before_external_calls(
    monkeypatch,
) -> None:
    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append((uri, self))

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
    )

    async def run_set() -> None:
        await cache.set(
            "recipes",
            "global",
            "番茄炒蛋",
            [
                Document(
                    page_content="# 番茄炒蛋",
                    metadata={
                        "parent_id": "doc-1",
                        "dish_name": "番茄炒蛋",
                        "category": "素菜",
                        "difficulty": "简单",
                        "source": "vegetable_dish/番茄炒蛋.md",
                        "data_source": 123,
                        "source_type": "recipes",
                        "user_id": "GLOBAL",
                    },
                )
            ],
            ttl=3600,
        )

    with pytest.raises(ValueError, match="retrieval cache document data_source must be text"):
        asyncio.run(run_set())

    assert created_clients == []


class NonFiniteDistanceMilvusClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def search(self, **kwargs):
        return [[{"distance": float("nan"), "entity": {"payload": self.payload}}]]


def test_milvus_vector_cache_treats_non_finite_distance_as_miss() -> None:
    cached_documents = [Document(page_content="# 番茄炒蛋", metadata={"parent_id": "doc-1"})]
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = NonFiniteDistanceMilvusClient(cache.encode_payload(cached_documents))

    result = asyncio.run(cache.get("recipes", "global", "番茄炒蛋", threshold=0.92))

    assert result is None


class MalformedHitMilvusClient:
    def search(self, **kwargs):
        return [["not a hit"]]


def test_milvus_vector_cache_treats_malformed_hit_as_miss() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = MalformedHitMilvusClient()

    result = asyncio.run(cache.get("recipes", "global", "番茄炒蛋", threshold=0.92))

    assert result is None


class MissingPayloadMilvusClient:
    def search(self, **kwargs):
        return [[{"distance": 0.99, "entity": {}}]]


def test_milvus_vector_cache_treats_missing_payload_as_miss() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = MissingPayloadMilvusClient()

    result = asyncio.run(cache.get("recipes", "global", "番茄炒蛋", threshold=0.92))

    assert result is None


class MalformedPayloadMilvusClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def search(self, **kwargs):
        return [[{"distance": 0.99, "entity": {"payload": self.payload}}]]


def test_milvus_vector_cache_treats_non_document_payload_as_miss() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = MalformedPayloadMilvusClient(cache.encode_payload(["not a document"]))

    result = asyncio.run(cache.get("recipes", "global", "番茄炒蛋", threshold=0.92))

    assert result is None


def test_milvus_vector_cache_treats_malformed_document_payload_as_miss() -> None:
    cache = MilvusVectorCache(
        milvus_uri="http://localhost:19530",
        collection_name="cache",
        embedding_client=FakeLLMClient(),
        embedding_dim=3,
    )
    cache._client = MalformedPayloadMilvusClient(
        cache.encode_payload([Document(page_content=object(), metadata=["not metadata"])])
    )

    result = asyncio.run(cache.get("recipes", "global", "番茄炒蛋", threshold=0.92))

    assert result is None
