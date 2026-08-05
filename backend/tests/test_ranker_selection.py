import pytest

from app.rag.document import Document
from app.rag.pipeline.retrieval import MilvusHybridRetriever


class DummyEmbeddingClient:
    pass


def build_retriever() -> MilvusHybridRetriever:
    return MilvusHybridRetriever(
        milvus_uri="unused",
        embedding_client=DummyEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "recipes"},
    )


def recipe_chunk_metadata(
    parent_id: str = "doc-1",
    dish_name: str = "番茄炒蛋",
    source: str = "vegetable_dish/番茄炒蛋.md",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "dish_name": dish_name,
        "category": "素菜",
        "difficulty": "简单",
        "source": source,
        "data_source": "recipes",
        "source_type": "recipes",
        "user_id": "GLOBAL",
        "is_dish_index": False,
    }


def test_easy_to_cook_rewritten_query_prefers_bm25_weight() -> None:
    retriever = build_retriever()

    assert retriever.intelligent_ranker_selection("有哪些简单易做的川菜菜谱？") == (0.4, 0.6)


def test_recommendation_query_prefers_dense_weight() -> None:
    retriever = build_retriever()

    assert retriever.intelligent_ranker_selection("推荐几道适合晚餐的清淡菜") == (0.6, 0.4)


@pytest.mark.parametrize("milvus_uri", ["", "   ", 123])
def test_milvus_hybrid_retriever_rejects_invalid_milvus_uri(milvus_uri) -> None:
    with pytest.raises(ValueError, match="Milvus URI must be text"):
        MilvusHybridRetriever(
            milvus_uri=milvus_uri,
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=3,
            collection_by_source={"recipes": "recipes"},
        )


@pytest.mark.parametrize("embedding_dim", [0, -1, True, 1.5, "1024"])
def test_milvus_hybrid_retriever_rejects_invalid_embedding_dim(embedding_dim) -> None:
    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        MilvusHybridRetriever(
            milvus_uri="unused",
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=embedding_dim,
            collection_by_source={"recipes": "recipes"},
        )


def test_milvus_hybrid_retriever_rejects_unknown_collection_sources() -> None:
    with pytest.raises(ValueError, match="unknown Milvus source: web"):
        MilvusHybridRetriever(
            milvus_uri="unused",
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=3,
            collection_by_source={"web": "cook_hero_web"},
        )


def test_milvus_hybrid_retriever_rejects_blank_collection_names() -> None:
    with pytest.raises(ValueError, match="Milvus collection name for recipes is required"):
        MilvusHybridRetriever(
            milvus_uri="unused",
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=3,
            collection_by_source={"recipes": "   "},
        )


def test_milvus_hybrid_retriever_rejects_unknown_ranker_strategy() -> None:
    with pytest.raises(ValueError, match="ranker_strategy must be one of: weighted, rrf"):
        MilvusHybridRetriever(
            milvus_uri="unused",
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=3,
            collection_by_source={"recipes": "recipes"},
            ranker_strategy="dense_only",
        )


@pytest.mark.parametrize("rrf_k", [0, -1, True, 1.5, "60"])
def test_milvus_hybrid_retriever_rejects_invalid_rrf_k(rrf_k) -> None:
    with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
        MilvusHybridRetriever(
            milvus_uri="unused",
            embedding_client=DummyEmbeddingClient(),
            embedding_dim=3,
            collection_by_source={"recipes": "recipes"},
            ranker_strategy="rrf",
            rrf_k=rrf_k,
        )


def test_milvus_text_field_accepts_long_markdown_chunks(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_schemas = []

    class FakeSchema:
        def __init__(self) -> None:
            self.fields = {}
            self.functions = []

        def add_field(self, name, data_type, **kwargs):
            self.fields[name] = {"data_type": data_type, **kwargs}

        def add_function(self, function):
            self.functions.append(function)

    class FakeIndexParams:
        def add_index(self, *args, **kwargs):
            return None

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            self.uri = uri

        @staticmethod
        def create_schema(auto_id: bool, enable_dynamic_field: bool):
            schema = FakeSchema()
            created_schemas.append(schema)
            return schema

        def has_collection(self, collection_name: str) -> bool:
            return False

        def prepare_index_params(self) -> FakeIndexParams:
            return FakeIndexParams()

        def create_collection(self, **kwargs):
            return None

    fake_pymilvus = SimpleNamespace(
        DataType=SimpleNamespace(
            INT64="INT64",
            VARCHAR="VARCHAR",
            FLOAT_VECTOR="FLOAT_VECTOR",
            SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
            BOOL="BOOL",
        ),
        Function=lambda **kwargs: kwargs,
        FunctionType=SimpleNamespace(BM25="BM25"),
        MilvusClient=FakeMilvusClient,
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    retriever = build_retriever()

    retriever.create_collections()

    assert created_schemas[0].fields["text"]["max_length"] == 65535
    assert created_schemas[0].fields["text"]["enable_analyzer"] is True


@pytest.mark.asyncio
async def test_milvus_index_rejects_non_list_documents_before_external_calls(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("invalid Milvus document batch should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="Milvus documents must be a list"):
        await retriever.index_documents("recipes", None)

    assert created_clients == []


def test_milvus_delete_rejects_non_list_documents_before_external_calls(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=DummyEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="Milvus documents must be a list"):
        retriever.delete_documents("recipes", None)

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_deletes_existing_parent_chunks_before_insert(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class EmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            self.uri = uri
            self.deleted_filters = []
            self.inserted_rows = []
            created_clients.append(self)

        def delete(self, *, collection_name: str, filter: str) -> None:
            self.deleted_filters.append((collection_name, filter))

        def insert(self, *, collection_name: str, data: list[dict]) -> None:
            self.inserted_rows.extend(data)

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=EmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    await retriever.index_documents(
        "recipes",
        [
            Document("# 第一段", recipe_chunk_metadata()),
            Document("## 第二段", recipe_chunk_metadata()),
        ],
    )

    client = created_clients[0]
    assert client.deleted_filters == [("cook_hero_recipes", 'parent_id == "doc-1"')]
    assert len(client.inserted_rows) == 2


@pytest.mark.asyncio
async def test_milvus_index_rejects_non_text_source_before_external_calls(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("invalid Milvus source should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="unknown Milvus source"):
        await retriever.index_documents(
            ["recipes"],
            [Document("# 番茄炒蛋", recipe_chunk_metadata())],
        )

    assert created_clients == []


def test_milvus_delete_rejects_documents_without_parent_id_before_external_calls(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

        def delete(self, **kwargs):
            raise AssertionError("invalid Milvus delete rows should not call delete")

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=DummyEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    metadata = recipe_chunk_metadata()
    metadata["parent_id"] = "   "

    with pytest.raises(ValueError, match="Milvus delete parent_id is required"):
        retriever.delete_documents("recipes", [Document("# 番茄炒蛋", metadata)])

    assert created_clients == []


def test_milvus_delete_rejects_unknown_source_before_external_calls(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

        def delete(self, **kwargs):
            raise AssertionError("invalid Milvus source should not call delete")

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=DummyEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="unknown Milvus source: web"):
        retriever.delete_documents(
            "web",
            [Document("# 番茄炒蛋", recipe_chunk_metadata())],
        )

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_rejects_rows_that_exceed_schema_limits_before_embedding(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("invalid Milvus rows should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="Milvus field source exceeds max length 1024"):
        await retriever.index_documents(
            "recipes",
            [Document("# 菜谱", recipe_chunk_metadata(source="s" * 1025))],
        )
    with pytest.raises(ValueError, match="Milvus field text exceeds max length 65535"):
        await retriever.index_documents(
            "recipes",
            [Document("x" * 65536, recipe_chunk_metadata(source="ok.md"))],
        )

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_rejects_non_boolean_is_dish_index_before_embedding(
    monkeypatch,
) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("invalid Milvus rows should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(
        sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient)
    )
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )
    metadata = recipe_chunk_metadata()
    metadata["is_dish_index"] = "false"

    with pytest.raises(ValueError, match="Milvus field is_dish_index must be a boolean"):
        await retriever.index_documents("recipes", [Document("# 番茄炒蛋", metadata)])

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_rejects_missing_source_type_before_embedding(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("invalid Milvus rows should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match="source_type"):
        await retriever.index_documents(
            "recipes",
            [
                Document(
                    "# 番茄炒蛋",
                    {
                        "parent_id": "doc-1",
                        "dish_name": "番茄炒蛋",
                        "category": "素菜",
                        "difficulty": "简单",
                        "source": "vegetable_dish/番茄炒蛋.md",
                        "data_source": "recipes",
                        "user_id": "GLOBAL",
                    },
                )
            ],
        )

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_rejects_embedding_vectors_with_wrong_dimension(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class WrongDimensionEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

        def delete(self, *, collection_name: str, filter: str) -> None:
            raise AssertionError("delete should not run after invalid embeddings")

        def insert(self, *, collection_name: str, data: list[dict]) -> None:
            raise AssertionError("insert should not run after invalid embeddings")

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=WrongDimensionEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(RuntimeError, match="embedding vector dimension mismatch"):
        await retriever.index_documents(
            "recipes",
            [Document("# 第一段", recipe_chunk_metadata())],
        )

    assert created_clients == []


@pytest.mark.asyncio
async def test_milvus_index_keeps_existing_chunks_when_embedding_fails(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class FailingEmbeddingClient:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding unavailable")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            self.uri = uri
            self.deleted_filters = []
            self.inserted_rows = []
            created_clients.append(self)

        def delete(self, *, collection_name: str, filter: str) -> None:
            self.deleted_filters.append((collection_name, filter))

        def insert(self, *, collection_name: str, data: list[dict]) -> None:
            self.inserted_rows.extend(data)

    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=FailingEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await retriever.index_documents(
            "recipes",
            [Document("# 第一段", recipe_chunk_metadata())],
        )

    assert created_clients == []


@pytest.mark.asyncio
async def test_hybrid_search_skips_hits_with_non_finite_distance(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    class EmbeddingClient:
        async def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    def entity(name: str) -> dict:
        return {
            "text": f"# {name}",
            "source": f"vegetable_dish/{name}.md",
            "parent_id": f"doc-{name}",
            "dish_name": name,
            "category": "素菜",
            "difficulty": "简单",
            "is_dish_index": False,
            "data_source": "recipes",
            "user_id": "GLOBAL",
            "source_type": "recipes",
        }

    class FakeMilvusClient:
        def hybrid_search(self, **kwargs):
            return [
                [
                    {"entity": entity("坏分数菜谱"), "distance": float("nan")},
                    {"entity": entity("番茄炒蛋"), "distance": 0.87},
                ]
            ]

    fake_pymilvus = SimpleNamespace(
        AnnSearchRequest=lambda **kwargs: kwargs,
        RRFRanker=lambda *args, **kwargs: ("rrf", args, kwargs),
        WeightedRanker=lambda *args, **kwargs: ("weighted", args, kwargs),
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=EmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )
    retriever._client = FakeMilvusClient()

    documents = await retriever.hybrid_search(
        source_name="recipes",
        query="番茄炒蛋怎么做",
        expr=None,
        top_k=2,
        fetch_multiplier=1,
    )

    assert [document.metadata["dish_name"] for document in documents] == ["番茄炒蛋"]
    assert documents[0].metadata["retrieval_score"] == 0.87


@pytest.mark.asyncio
async def test_hybrid_search_skips_hits_missing_required_entity_fields(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    class EmbeddingClient:
        async def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    complete_entity = {
        "text": "# 番茄炒蛋",
        "source": "vegetable_dish/番茄炒蛋.md",
        "parent_id": "doc-tomato-egg",
        "dish_name": "番茄炒蛋",
        "category": "素菜",
        "difficulty": "简单",
        "is_dish_index": False,
        "data_source": "recipes",
        "user_id": "GLOBAL",
        "source_type": "recipes",
    }

    class FakeMilvusClient:
        def hybrid_search(self, **kwargs):
            return [
                [
                    {
                        "entity": {
                            "text": "# 缺父文档菜谱",
                            "source": "vegetable_dish/missing-parent.md",
                            "dish_name": "缺父文档菜谱",
                            "category": "素菜",
                            "difficulty": "简单",
                            "is_dish_index": False,
                            "data_source": "recipes",
                            "user_id": "GLOBAL",
                            "source_type": "recipes",
                        },
                        "distance": 0.99,
                    },
                    {"entity": complete_entity, "distance": 0.87},
                ]
            ]

    fake_pymilvus = SimpleNamespace(
        AnnSearchRequest=lambda **kwargs: kwargs,
        RRFRanker=lambda *args, **kwargs: ("rrf", args, kwargs),
        WeightedRanker=lambda *args, **kwargs: ("weighted", args, kwargs),
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=EmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )
    retriever._client = FakeMilvusClient()

    documents = await retriever.hybrid_search(
        source_name="recipes",
        query="番茄炒蛋怎么做",
        expr=None,
        top_k=2,
        fetch_multiplier=1,
    )

    assert [document.metadata["parent_id"] for document in documents] == ["doc-tomato-egg"]
    assert documents[0].page_content == "# 番茄炒蛋"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"source_name": "web", "query": "番茄炒蛋", "expr": None, "top_k": 2, "fetch_multiplier": 1},
            "unknown Milvus source",
        ),
        (
            {"source_name": "recipes", "query": "   ", "expr": None, "top_k": 2, "fetch_multiplier": 1},
            "query must be a non-empty string",
        ),
        (
            {"source_name": "recipes", "query": 123, "expr": None, "top_k": 2, "fetch_multiplier": 1},
            "query must be a non-empty string",
        ),
        (
            {"source_name": "recipes", "query": "番茄炒蛋", "expr": 123, "top_k": 2, "fetch_multiplier": 1},
            "expr must be a string or None",
        ),
        (
            {"source_name": "recipes", "query": "番茄炒蛋", "expr": None, "top_k": True, "fetch_multiplier": 1},
            "top_k must be positive",
        ),
        (
            {"source_name": "recipes", "query": "番茄炒蛋", "expr": None, "top_k": 0, "fetch_multiplier": 1},
            "top_k must be positive",
        ),
        (
            {"source_name": "recipes", "query": "番茄炒蛋", "expr": None, "top_k": 2, "fetch_multiplier": 0},
            "fetch_multiplier must be positive",
        ),
    ],
)
async def test_hybrid_search_rejects_invalid_inputs_before_external_calls(monkeypatch, kwargs, message) -> None:
    import sys
    from types import SimpleNamespace

    created_clients = []

    class NoCallEmbeddingClient:
        async def embed_query(self, text: str) -> list[float]:
            raise AssertionError("invalid hybrid search inputs should not call embedding API")

    class FakeMilvusClient:
        def __init__(self, uri: str) -> None:
            created_clients.append(self)

        def hybrid_search(self, **kwargs):
            raise AssertionError("invalid hybrid search inputs should not query Milvus")

    fake_pymilvus = SimpleNamespace(
        AnnSearchRequest=lambda **kwargs: kwargs,
        MilvusClient=FakeMilvusClient,
        RRFRanker=lambda *args, **kwargs: ("rrf", args, kwargs),
        WeightedRanker=lambda *args, **kwargs: ("weighted", args, kwargs),
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    retriever = MilvusHybridRetriever(
        milvus_uri="http://localhost:19530",
        embedding_client=NoCallEmbeddingClient(),
        embedding_dim=3,
        collection_by_source={"recipes": "cook_hero_recipes"},
    )

    with pytest.raises(ValueError, match=message):
        await retriever.hybrid_search(**kwargs)

    assert created_clients == []
