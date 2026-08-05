from __future__ import annotations

import math

from app.core.llm import OpenAICompatibleClient
from app.rag.document import Document
from app.rag.milvus_expr import milvus_string
from app.rag.vector_validation import validate_embedding_vector, validate_embedding_vectors


MILVUS_VARCHAR_LIMITS = {
    "text": 65535,
    "source": 1024,
    "parent_id": 128,
    "dish_name": 256,
    "category": 128,
    "difficulty": 64,
    "data_source": 64,
    "user_id": 128,
    "source_type": 64,
}
MILVUS_RETRIEVAL_STRING_FIELDS = tuple(MILVUS_VARCHAR_LIMITS)
MILVUS_RETRIEVAL_OUTPUT_FIELDS = [
    "text",
    "source",
    "parent_id",
    "dish_name",
    "category",
    "difficulty",
    "is_dish_index",
    "data_source",
    "user_id",
    "source_type",
]
ALLOWED_MILVUS_SOURCES = {"recipes", "personal"}


class MilvusHybridRetriever:
    def __init__(
        self,
        milvus_uri: str,
        embedding_client: OpenAICompatibleClient,
        embedding_dim: int,
        collection_by_source: dict[str, str],
        ranker_strategy: str = "weighted",
        rrf_k: int = 60,
    ) -> None:
        if not isinstance(milvus_uri, str) or not milvus_uri.strip():
            raise ValueError("Milvus URI must be text")
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(ranker_strategy, str):
            raise ValueError("ranker_strategy must be one of: weighted, rrf")
        normalized_ranker_strategy = ranker_strategy.strip().lower()
        if normalized_ranker_strategy not in {"weighted", "rrf"}:
            raise ValueError("ranker_strategy must be one of: weighted, rrf")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if not isinstance(collection_by_source, dict) or not collection_by_source:
            raise ValueError("Milvus collection_by_source must not be empty")
        normalized_collections: dict[str, str] = {}
        for source_name, collection_name in collection_by_source.items():
            if source_name not in ALLOWED_MILVUS_SOURCES:
                raise ValueError(f"unknown Milvus source: {source_name}")
            if not isinstance(collection_name, str) or not collection_name.strip():
                raise ValueError(f"Milvus collection name for {source_name} is required")
            normalized_collections[source_name] = collection_name.strip()
        self.milvus_uri = milvus_uri.strip()
        self._client = None
        self.embedding_client = embedding_client
        self.embedding_dim = embedding_dim
        self.collection_by_source = normalized_collections
        self.ranker_strategy = normalized_ranker_strategy
        self.rrf_k = rrf_k

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.milvus_uri)
        return self._client

    def create_collections(self) -> None:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        for collection_name in self.collection_by_source.values():
            if self.client.has_collection(collection_name):
                continue
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("pk", DataType.INT64, is_primary=True)
            schema.add_field("text", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["text"], enable_analyzer=True)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("source", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["source"])
            schema.add_field("parent_id", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["parent_id"])
            schema.add_field("dish_name", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["dish_name"])
            schema.add_field("category", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["category"])
            schema.add_field("difficulty", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["difficulty"])
            schema.add_field("is_dish_index", DataType.BOOL)
            schema.add_field("data_source", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["data_source"])
            schema.add_field("user_id", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["user_id"])
            schema.add_field("source_type", DataType.VARCHAR, max_length=MILVUS_VARCHAR_LIMITS["source_type"])
            bm25_function = Function(
                name="bm25_function",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)
            index_params = self.client.prepare_index_params()
            index_params.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index("sparse", index_type="AUTOINDEX", metric_type="BM25")
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        if not isinstance(documents, list):
            raise ValueError("Milvus documents must be a list")
        if not documents:
            return
        if not isinstance(source_name, str) or source_name not in self.collection_by_source:
            raise ValueError(f"unknown Milvus source: {source_name}")
        collection_name = self.collection_by_source[source_name]
        rows = [self._row_without_vector(document) for document in documents]
        self._validate_milvus_rows(rows)
        self._validate_milvus_source_scope(source_name, rows)
        vectors = validate_embedding_vectors(
            await self.embedding_client.embed_documents([row["text"] for row in rows]),
            self.embedding_dim,
        )
        rows_with_vectors = [
            {**row, "dense": vector}
            for row, vector in zip(rows, vectors, strict=True)
        ]
        self._delete_existing_parent_chunks(collection_name, documents)
        self.client.insert(collection_name=collection_name, data=rows_with_vectors)

    @staticmethod
    def _row_without_vector(document: Document) -> dict[str, object]:
        return {
            "text": document.page_content,
            "source": document.metadata.get("source"),
            "parent_id": document.metadata.get("parent_id"),
            "dish_name": document.metadata.get("dish_name"),
            "category": document.metadata.get("category"),
            "difficulty": document.metadata.get("difficulty"),
            "is_dish_index": document.metadata.get("is_dish_index", False),
            "data_source": document.metadata.get("data_source"),
            "user_id": document.metadata.get("user_id"),
            "source_type": document.metadata.get("source_type"),
        }

    @staticmethod
    def _validate_milvus_rows(rows: list[dict[str, object]]) -> None:
        for row in rows:
            for field, max_length in MILVUS_VARCHAR_LIMITS.items():
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Milvus field {field} is required")
                if len(value.encode("utf-8")) > max_length:
                    raise ValueError(f"Milvus field {field} exceeds max length {max_length}")
            if not isinstance(row.get("is_dish_index"), bool):
                raise ValueError("Milvus field is_dish_index must be a boolean")

    @staticmethod
    def _validate_milvus_source_scope(source_name: str, rows: list[dict[str, object]]) -> None:
        for row in rows:
            data_source = str(row.get("data_source") or "")
            source_type = str(row.get("source_type") or "")
            user_id = str(row.get("user_id") or "")
            if data_source != source_name:
                raise ValueError("Milvus field data_source must match source_name")
            if source_type != source_name:
                raise ValueError("Milvus field source_type must match source_name")
            if source_name == "recipes" and user_id != "GLOBAL":
                raise ValueError("Milvus recipe rows must use GLOBAL user_id")
            if source_name == "personal" and (not user_id or user_id == "GLOBAL"):
                raise ValueError("Milvus personal rows require a non-GLOBAL user_id")

    @staticmethod
    def _clean_retrieval_entity(entity: object) -> dict[str, object] | None:
        if not isinstance(entity, dict):
            return None
        cleaned: dict[str, object] = {}
        for field in MILVUS_RETRIEVAL_STRING_FIELDS:
            value = entity.get(field)
            if not isinstance(value, str):
                return None
            value = value.strip()
            if not value:
                return None
            cleaned[field] = value
        is_dish_index = entity.get("is_dish_index")
        if not isinstance(is_dish_index, bool):
            return None
        cleaned["is_dish_index"] = is_dish_index
        return cleaned

    def delete_documents(self, source_name: str, documents: list[Document]) -> None:
        if not isinstance(documents, list):
            raise ValueError("Milvus documents must be a list")
        if not documents:
            return
        if not isinstance(source_name, str) or source_name not in self.collection_by_source:
            raise ValueError(f"unknown Milvus source: {source_name}")
        collection_name = self.collection_by_source[source_name]
        self._delete_existing_parent_chunks(collection_name, documents)

    def _delete_existing_parent_chunks(self, collection_name: str, documents: list[Document]) -> None:
        parent_ids: set[str] = set()
        for document in documents:
            parent_id = document.metadata.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise ValueError("Milvus delete parent_id is required")
            parent_ids.add(parent_id.strip())
        for parent_id in sorted(parent_ids):
            self.client.delete(
                collection_name=collection_name,
                filter=f"parent_id == {milvus_string(parent_id)}",
            )

    def intelligent_ranker_selection(self, query: str) -> tuple[float, float]:
        if any(keyword in query for keyword in ("怎么做", "步骤", "做法", "制作", "易做")):
            return 0.4, 0.6
        if any(keyword in query for keyword in ("推荐", "类似", "适合", "建议", "来点")):
            return 0.6, 0.4
        return 0.5, 0.5

    async def hybrid_search(
        self,
        source_name: str,
        query: str,
        expr: str | None,
        top_k: int,
        fetch_multiplier: int,
    ) -> list[Document]:
        if not isinstance(source_name, str) or source_name not in self.collection_by_source:
            raise ValueError(f"unknown Milvus source: {source_name}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        if expr is not None and not isinstance(expr, str):
            raise ValueError("expr must be a string or None")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        if isinstance(fetch_multiplier, bool) or not isinstance(fetch_multiplier, int) or fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be positive")

        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        collection_name = self.collection_by_source[source_name]
        dense_weight, sparse_weight = self.intelligent_ranker_selection(query)
        limit = max(top_k * fetch_multiplier, top_k)
        dense_vector = validate_embedding_vector(
            await self.embedding_client.embed_query(query), self.embedding_dim
        )
        requests = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense",
                param={"metric_type": "COSINE"},
                limit=limit,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field="sparse",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=expr,
            ),
        ]
        ranker = (
            RRFRanker(self.rrf_k)
            if self.ranker_strategy.lower() == "rrf"
            else WeightedRanker(dense_weight, sparse_weight)
        )
        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=requests,
            ranker=ranker,
            limit=limit,
            output_fields=MILVUS_RETRIEVAL_OUTPUT_FIELDS,
        )
        documents: list[Document] = []
        for hit in results[0] if results else []:
            try:
                retrieval_score = float(hit.get("distance", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(retrieval_score):
                continue
            entity = self._clean_retrieval_entity(hit.get("entity"))
            if entity is None:
                continue
            documents.append(
                Document(
                    page_content=str(entity["text"]),
                    metadata={
                        "source": entity["source"],
                        "parent_id": entity["parent_id"],
                        "dish_name": entity["dish_name"],
                        "category": entity["category"],
                        "difficulty": entity["difficulty"],
                        "is_dish_index": entity["is_dish_index"],
                        "data_source": entity["data_source"],
                        "user_id": entity["user_id"],
                        "source_type": entity["source_type"],
                        "retrieval_score": retrieval_score,
                        "ranker_weights": [dense_weight, sparse_weight],
                        "ranker_strategy": self.ranker_strategy,
                    },
                )
            )
        return documents


RetrievalOptimizationModule = MilvusHybridRetriever
