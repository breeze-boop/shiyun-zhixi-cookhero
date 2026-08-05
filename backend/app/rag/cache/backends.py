from __future__ import annotations

import base64
import hashlib
import math
import pickle
import time
import zlib
from dataclasses import dataclass

from app.core.llm import OpenAICompatibleClient
from app.rag.document import Document
from app.rag.milvus_expr import milvus_string
from app.rag.vector_validation import validate_embedding_vector


MILVUS_CACHE_VARCHAR_LIMITS = {
    "source": 64,
    "scope": 128,
    "query": 2048,
    "payload": 65535,
}


CACHE_IDENTITY_FIELDS = {"source", "scope", "query"}
DOCUMENT_IDENTITY_METADATA_FIELDS = (
    "parent_id",
    "dish_name",
    "category",
    "difficulty",
    "source",
    "data_source",
    "source_type",
    "user_id",
)


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _normalize_cache_identity(source: str, scope: str, query: str) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field, raw_value in {"source": source, "scope": scope, "query": query}.items():
        if not isinstance(raw_value, str):
            raise ValueError(f"retrieval cache field {field} must be text")
        value = raw_value.strip()
        if not value:
            raise ValueError(f"retrieval cache field {field} is required")
        identity[field] = value
    return identity


def _normalize_cache_ttl(ttl: int) -> int:
    if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
        raise ValueError("retrieval cache ttl must be positive")
    return ttl


def _normalize_cache_threshold(threshold: float) -> float:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise ValueError("Milvus cache threshold must be between 0 and 1")
    return float(threshold)


def _validate_milvus_cache_row(row: dict[str, str]) -> None:
    for field, max_length in MILVUS_CACHE_VARCHAR_LIMITS.items():
        value = row[field]
        if field not in CACHE_IDENTITY_FIELDS and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Milvus cache field {field} is required")
        if _utf8_len(value) > max_length:
            raise ValueError(
                f"Milvus cache field {field} exceeds max length {max_length}"
            )


def _validate_document_payload(value: object) -> list[Document]:
    if not isinstance(value, list):
        raise ValueError("retrieval cache payload must be a list of Document")
    for document in value:
        if not isinstance(document, Document):
            raise ValueError("retrieval cache payload must be a list of Document")
        if not isinstance(document.page_content, str) or not document.page_content.strip():
            raise ValueError("retrieval cache document content must be text")
        if not isinstance(document.metadata, dict):
            raise ValueError("retrieval cache document metadata must be an object")
        for field in DOCUMENT_IDENTITY_METADATA_FIELDS:
            if field in document.metadata and not isinstance(document.metadata[field], str):
                raise ValueError(f"retrieval cache document {field} must be text")
    return value


@dataclass(slots=True)
class CacheEntry:
    value: list[Document]
    expires_at: float


class KeywordCacheBackend:
    async def get(self, source: str, scope: str, query: str) -> list[Document] | None:
        raise NotImplementedError

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        raise NotImplementedError


class VectorCacheBackend:
    async def get(
        self, source: str, scope: str, query: str, threshold: float
    ) -> list[Document] | None:
        raise NotImplementedError

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        raise NotImplementedError


class RedisKeywordCache(KeywordCacheBackend):
    def __init__(self, redis_url: str) -> None:
        if not isinstance(redis_url, str) or not redis_url.strip():
            raise ValueError("Redis cache URL must be text")
        import redis

        self.client = redis.Redis.from_url(redis_url.strip())

    async def get(self, source: str, scope: str, query: str) -> list[Document] | None:
        identity = _normalize_cache_identity(source, scope, query)
        payload = self.client.get(self._key(**identity))
        if payload is None:
            return None
        return _validate_document_payload(pickle.loads(payload))

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        documents = _validate_document_payload(documents)
        cache_ttl = _normalize_cache_ttl(ttl)
        identity = _normalize_cache_identity(source, scope, query)
        self.client.setex(self._key(**identity), cache_ttl, pickle.dumps(documents))

    @staticmethod
    def _key(source: str, scope: str, query: str) -> str:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"rag:retrieval:{source}:{scope}:{digest}"


class MilvusVectorCache(VectorCacheBackend):
    def __init__(
        self,
        milvus_uri: str,
        collection_name: str,
        embedding_client: OpenAICompatibleClient,
        embedding_dim: int,
    ) -> None:
        if not isinstance(milvus_uri, str) or not milvus_uri.strip():
            raise ValueError("Milvus cache URI must be text")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Milvus cache collection name must be text")
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise ValueError("Milvus cache embedding_dim must be a positive integer")
        self.milvus_uri = milvus_uri.strip()
        self._client = None
        self.collection_name = collection_name.strip()
        self.embedding_client = embedding_client
        self.embedding_dim = embedding_dim

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.milvus_uri)
        return self._client

    @staticmethod
    def encode_payload(documents: list[Document]) -> str:
        compressed = zlib.compress(pickle.dumps(documents), level=9)
        return base64.b64encode(compressed).decode("ascii")

    @staticmethod
    def decode_payload(payload: str) -> list[Document]:
        compressed = base64.b64decode(payload.encode("ascii"))
        return _validate_document_payload(pickle.loads(zlib.decompress(compressed)))

    def create_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        if self.client.has_collection(self.collection_name):
            return
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(
            "source", DataType.VARCHAR, max_length=MILVUS_CACHE_VARCHAR_LIMITS["source"]
        )
        schema.add_field(
            "scope", DataType.VARCHAR, max_length=MILVUS_CACHE_VARCHAR_LIMITS["scope"]
        )
        schema.add_field(
            "query", DataType.VARCHAR, max_length=MILVUS_CACHE_VARCHAR_LIMITS["query"]
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(
            "payload", DataType.VARCHAR, max_length=MILVUS_CACHE_VARCHAR_LIMITS["payload"]
        )
        schema.add_field("expires_at", DataType.INT64)
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)

    async def get(
        self, source: str, scope: str, query: str, threshold: float
    ) -> list[Document] | None:
        cache_threshold = _normalize_cache_threshold(threshold)
        identity = _normalize_cache_identity(source, scope, query)
        vector = validate_embedding_vector(
            await self.embedding_client.embed_query(identity["query"]), self.embedding_dim
        )
        now = int(time.time())
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=1,
            filter=(
                f"source == {milvus_string(identity['source'])} "
                f"and scope == {milvus_string(identity['scope'])} "
                f"and expires_at >= {now}"
            ),
            output_fields=["payload"],
        )
        if not results or not results[0]:
            return None
        hit = results[0][0]
        if not isinstance(hit, dict):
            return None
        try:
            distance = float(hit.get("distance", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance) or distance < cache_threshold:
            return None
        entity = hit.get("entity")
        if not isinstance(entity, dict):
            return None
        payload = entity.get("payload")
        if not isinstance(payload, str) or not payload:
            return None
        try:
            return self.decode_payload(payload)
        except Exception:
            return None

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        documents = _validate_document_payload(documents)
        cache_ttl = _normalize_cache_ttl(ttl)
        row = {
            **_normalize_cache_identity(source, scope, query),
            "payload": self.encode_payload(documents),
            "expires_at": int(time.time()) + cache_ttl,
        }
        _validate_milvus_cache_row(row)
        vector = validate_embedding_vector(
            await self.embedding_client.embed_query(row["query"]), self.embedding_dim
        )
        self.client.insert(
            collection_name=self.collection_name,
            data=[{**row, "vector": vector}],
        )
