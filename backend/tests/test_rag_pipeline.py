from pathlib import Path

import pytest

from app.agent.tools import KnowledgeBaseSearchTool
from app.rag.document import Document, ParsedDocument
from app.services.rag_service import SourceRetrievalBatch
from scripts.howtocook_loader import HowToCookLoader
from tests.fakes import FakeLLMClient, build_test_rag_service


@pytest.mark.asyncio
async def test_rag_pipeline_rewrites_filters_retrieves_and_restores_parent() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    result = await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=3)

    assert "详细制作步骤" in result.rewritten_query
    assert result.metadata_expression == 'dish_name == "番茄炒蛋"'
    assert result.documents
    assert result.documents[0].metadata["restored_parent"] is True
    assert "# 番茄炒蛋" in result.context
    assert "## 必备原料和工具" in result.context


@pytest.mark.asyncio
async def test_rag_pipeline_uses_cache_on_repeat_query() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    await service.retrieve("推荐简单的汤", sources=["recipes"], top_k=3)
    second = await service.retrieve("推荐简单的汤", sources=["recipes"], top_k=3)

    assert any(item.startswith("cache_hit:recipes:L1") for item in second.trace)


@pytest.mark.asyncio
async def test_rag_pipeline_writes_restored_parent_documents_to_cache() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    first = await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=3)
    cached = await service.cache_manager.get("recipes", first.rewritten_query, "global")

    assert cached is not None
    _, documents = cached
    assert documents[0].metadata["restored_parent"] is True
    assert "# 番茄炒蛋" in documents[0].page_content
    assert "## 必备原料和工具" in documents[0].page_content


@pytest.mark.asyncio
async def test_corrupt_cache_hit_falls_back_to_fresh_retrieval() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="healthy-cache-fallback-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            )
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "健康晚餐",
        [
            Document(
                page_content="# 损坏缓存",
                metadata={
                    "parent_id": "missing-cache-parent",
                    "dish_name": "损坏缓存",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "cache/stale.md",
                    "data_source": "recipes",
                    "source_type": "recipes",
                    "user_id": "GLOBAL",
                },
            )
        ],
        ttl=3600,
    )

    result = await service.retrieve("健康晚餐", sources=["recipes"], top_k=1)

    assert "健康晚餐菜谱" in result.context
    assert result.documents[0].metadata["restored_parent"] is True
    assert "cache_rejected:recipes:restore" in result.trace
    assert any(item.startswith("hybrid:recipes") for item in result.trace)


@pytest.mark.asyncio
async def test_cache_hit_rejects_documents_from_wrong_source_scope() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="recipe-cache-scope-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            )
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "健康晚餐",
        [
            Document(
                page_content="# 用户二私密偏好\n\n不要返回这条个人缓存。",
                metadata={
                    "parent_id": "personal-u2-cache-doc",
                    "dish_name": "用户二私密偏好",
                    "category": "个人饮食",
                    "difficulty": "普通",
                    "source": "personal/u2/用户二私密偏好.md",
                    "data_source": "personal",
                    "source_type": "personal",
                    "user_id": "u2",
                    "restored_parent": True,
                    "retrieval_score": 0.99,
                },
            )
        ],
        ttl=3600,
    )

    result = await service.retrieve("健康晚餐", sources=["recipes"], top_k=1)

    assert "健康晚餐菜谱" in result.context
    assert "用户二私密偏好" not in result.context
    assert result.sources[0].data_source == "recipes"
    assert "cache_rejected:recipes:scope" in result.trace
    assert any(item.startswith("hybrid:recipes") for item in result.trace)


@pytest.mark.asyncio
async def test_cache_hit_rejects_documents_missing_source_type() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="recipe-cache-missing-source-type-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            )
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "健康晚餐",
        [
            Document(
                page_content="# 缺来源类型缓存\n\n不要返回这条不完整缓存。",
                metadata={
                    "parent_id": "recipe-cache-missing-source-type-doc",
                    "dish_name": "缺来源类型缓存",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "cache/missing-source-type.md",
                    "data_source": "recipes",
                    "user_id": "GLOBAL",
                    "restored_parent": True,
                    "retrieval_score": 0.99,
                },
            )
        ],
        ttl=3600,
    )

    result = await service.retrieve("健康晚餐", sources=["recipes"], top_k=1)

    assert "健康晚餐菜谱" in result.context
    assert "缺来源类型缓存" not in result.context
    assert "cache_rejected:recipes:scope" in result.trace
    assert any(item.startswith("hybrid:recipes") for item in result.trace)


@pytest.mark.asyncio
async def test_cache_hit_rejects_restored_documents_with_blank_source_identity() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="blank-cache-source-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            )
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "健康晚餐",
        [
            Document(
                page_content="# 坏来源缓存\n\n不要返回这条没有来源身份的缓存。",
                metadata={
                    "parent_id": "blank-cache-source-doc",
                    "dish_name": "   ",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "   ",
                    "data_source": "recipes",
                    "source_type": "recipes",
                    "user_id": "GLOBAL",
                    "restored_parent": True,
                    "retrieval_score": 0.99,
                },
            )
        ],
        ttl=3600,
    )

    result = await service.retrieve("健康晚餐", sources=["recipes"], top_k=1)

    assert "健康晚餐菜谱" in result.context
    assert "坏来源缓存" not in result.context
    assert result.sources[0].title == "健康晚餐菜谱"
    assert result.sources[0].source == "vegetable_dish/健康晚餐菜谱.md"
    assert "cache_rejected:recipes:restore" in result.trace
    assert any(item.startswith("hybrid:recipes") for item in result.trace)


@pytest.mark.asyncio
async def test_malformed_cached_scores_fall_back_to_fresh_retrieval() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="malformed-cache-score-doc",
                dish_name="清淡晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 清淡晚餐菜谱\n\n清淡晚餐可以选择蒸蛋和青菜。",
                source="vegetable_dish/清淡晚餐菜谱.md",
            )
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "清淡晚餐",
        [
            Document(
                page_content="# 清淡晚餐菜谱\n\n旧缓存内容。",
                metadata={
                    "parent_id": "malformed-cache-score-doc",
                    "dish_name": "清淡晚餐菜谱",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "vegetable_dish/清淡晚餐菜谱.md",
                    "data_source": "recipes",
                    "source_type": "recipes",
                    "user_id": "GLOBAL",
                    "retrieval_score": "not-a-number",
                },
            )
        ],
        ttl=3600,
    )

    result = await service.retrieve("清淡晚餐", sources=["recipes"], top_k=1)

    assert "清淡晚餐菜谱" in result.context
    assert result.documents[0].metadata["restored_parent"] is True
    assert "cache_rejected:recipes:restore" in result.trace
    assert any(item.startswith("hybrid:recipes") for item in result.trace)


@pytest.mark.asyncio
async def test_knowledge_base_tool_serializes_slot_sources() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    payload = await KnowledgeBaseSearchTool(service).execute("番茄炒蛋怎么做")

    assert payload["sources"][0]["dish_name"] == "番茄炒蛋"


@pytest.mark.asyncio
async def test_personal_source_requires_user_id_for_retrieval() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="personal-doc-1",
                dish_name="训练日晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 训练日晚餐偏好\n\n偏好高蛋白、低油。",
                source="personal/u1/训练日晚餐偏好.md",
                data_source="personal",
                source_type="personal",
                user_id="u1",
            )
        ]
    )

    anonymous = await service.retrieve("训练日晚餐偏好", sources=["personal"], top_k=3)
    scoped = await service.retrieve("训练日晚餐偏好", user_id="u1", sources=["personal"], top_k=3)

    assert anonymous.documents == []
    assert "训练日晚餐偏好" in scoped.context


@pytest.mark.asyncio
async def test_multi_source_retrieval_writes_each_missed_source_cache() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="recipe-cache-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            ),
            ParsedDocument(
                doc_id="personal-cache-doc",
                dish_name="训练日晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 训练日晚餐偏好\n\n健康晚餐偏好高蛋白和少油。",
                source="personal/u1/训练日晚餐偏好.md",
                data_source="personal",
                source_type="personal",
                user_id="u1",
            ),
        ]
    )

    first = await service.retrieve("健康晚餐", user_id="u1", sources=["recipes", "personal"], top_k=1)
    second = await service.retrieve("健康晚餐", user_id="u1", sources=["recipes", "personal"], top_k=1)

    assert "cache_write:recipes" in first.trace
    assert "cache_write:personal" in first.trace
    assert any(item.startswith("cache_hit:recipes:L1") for item in second.trace)
    assert any(item.startswith("cache_hit:personal:L1") for item in second.trace)


@pytest.mark.asyncio
async def test_cache_write_keeps_personal_documents_in_current_scope() -> None:
    service = build_test_rag_service()
    rewritten_query = "健康晚餐推荐"
    trace: list[str] = []
    personal_batch = SourceRetrievalBatch(
        source_name="personal",
        scope="u1",
        documents=[],
        cache_hit=False,
    )
    restored_documents = [
        Document(
            page_content="# 用户一晚餐偏好\n\n少油高蛋白。",
            metadata={
                "parent_id": "personal-u1-doc",
                "dish_name": "用户一晚餐偏好",
                "category": "个人饮食",
                "difficulty": "普通",
                "source": "personal/u1/晚餐偏好.md",
                "data_source": "personal",
                "source_type": "personal",
                "user_id": "u1",
                "restored_parent": True,
                "retrieval_score": 0.9,
            },
        ),
        Document(
            page_content="# 用户二晚餐偏好\n\n不要写入用户一缓存。",
            metadata={
                "parent_id": "personal-u2-doc",
                "dish_name": "用户二晚餐偏好",
                "category": "个人饮食",
                "difficulty": "普通",
                "source": "personal/u2/晚餐偏好.md",
                "data_source": "personal",
                "source_type": "personal",
                "user_id": "u2",
                "restored_parent": True,
                "retrieval_score": 0.99,
            },
        ),
    ]

    await service._write_missed_source_caches([personal_batch], rewritten_query, restored_documents, trace)
    cached = await service.cache_manager.get("personal", rewritten_query, "u1")

    assert cached is not None
    _, documents = cached
    assert [document.metadata["user_id"] for document in documents] == ["u1"]
    assert "用户二晚餐偏好" not in documents[0].page_content


@pytest.mark.asyncio
async def test_cache_write_failure_degrades_without_success_trace() -> None:
    from app.rag.cache.cache_manager import CacheManager
    from app.rag.cache.backends import KeywordCacheBackend, VectorCacheBackend

    class FailingKeywordCache(KeywordCacheBackend):
        async def get(self, source: str, scope: str, query: str) -> list[Document] | None:
            raise RuntimeError("redis unavailable")

        async def set(
            self, source: str, scope: str, query: str, documents: list[Document], ttl: int
        ) -> None:
            raise RuntimeError("redis unavailable")

    class FailingVectorCache(VectorCacheBackend):
        async def get(
            self, source: str, scope: str, query: str, threshold: float
        ) -> list[Document] | None:
            raise RuntimeError("milvus cache unavailable")

        async def set(
            self, source: str, scope: str, query: str, documents: list[Document], ttl: int
        ) -> None:
            raise RuntimeError("milvus cache unavailable")

    service = build_test_rag_service()
    service.cache_manager = CacheManager(
        keyword_backend=FailingKeywordCache(),
        vector_backend=FailingVectorCache(),
        ttl_seconds=3600,
        l2_threshold=0.92,
    )
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="cache-write-failure-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            )
        ]
    )

    result = await service.retrieve("健康晚餐", sources=["recipes"], top_k=1)

    assert "健康晚餐菜谱" in result.context
    assert "cache_write:recipes" not in result.trace
    assert "cache_write_failed:recipes" in result.trace


class SameRewriteFilterLLM(FakeLLMClient):
    async def complete_text(self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.1) -> str:
        if "查询改写" in system_prompt:
            return "晚餐推荐"
        return await super().complete_text(system_prompt, user_prompt, model=model, temperature=temperature)

    async def complete_json(self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.0) -> dict:
        import json

        query = json.loads(user_prompt).get("query", "")
        if "汤" in query:
            return {"expr": "category == \"汤品\" and difficulty == \"简单\""}
        return {"expr": "NONE"}


class PercentLiteralMetadataLLM(FakeLLMClient):
    async def complete_text(self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.1) -> str:
        if "查询改写" in system_prompt:
            return "百分号菜名"
        return await super().complete_text(system_prompt, user_prompt, model=model, temperature=temperature)

    async def complete_json(self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.0) -> dict:
        if "Milvus 元数据自查询过滤器" in system_prompt:
            return {"expr": 'dish_name LIKE "%%折扣菜%"'}
        return await super().complete_json(system_prompt, user_prompt, model=model, temperature=temperature)


@pytest.mark.asyncio
async def test_cache_like_filter_preserves_percent_inside_dictionary_value() -> None:
    service = build_test_rag_service()
    service.llm_client = PercentLiteralMetadataLLM()
    service.generation.llm_client = service.llm_client
    service.metadata_filter.llm_client = service.llm_client
    service.repository.upsert_documents(
        [
            ParsedDocument(
                doc_id="percent-literal-doc",
                dish_name="%折扣菜",
                category="素菜",
                difficulty="普通",
                content="# %折扣菜\n\n这条菜名本身包含百分号。",
                source="vegetable_dish/percent-literal.md",
            ),
            ParsedDocument(
                doc_id="plain-discount-doc",
                dish_name="折扣菜",
                category="素菜",
                difficulty="普通",
                content="# 折扣菜\n\n这条不应该被百分号字面值过滤命中。",
                source="vegetable_dish/plain-discount.md",
            ),
        ]
    )
    await service.cache_manager.keyword_backend.set(
        "recipes",
        "global",
        "百分号菜名",
        [
            Document(
                page_content="# %折扣菜\n\n这条菜名本身包含百分号。",
                metadata={
                    "parent_id": "percent-literal-doc",
                    "dish_name": "%折扣菜",
                    "category": "素菜",
                    "difficulty": "普通",
                    "source": "vegetable_dish/percent-literal.md",
                    "data_source": "recipes",
                    "source_type": "recipes",
                    "user_id": "GLOBAL",
                    "restored_parent": True,
                    "retrieval_score": 0.9,
                },
            ),
            Document(
                page_content="# 折扣菜\n\n这条不应该被百分号字面值过滤命中。",
                metadata={
                    "parent_id": "plain-discount-doc",
                    "dish_name": "折扣菜",
                    "category": "素菜",
                    "difficulty": "普通",
                    "source": "vegetable_dish/plain-discount.md",
                    "data_source": "recipes",
                    "source_type": "recipes",
                    "user_id": "GLOBAL",
                    "restored_parent": True,
                    "retrieval_score": 0.8,
                },
            ),
        ],
        ttl=3600,
    )

    result = await service.retrieve("百分号菜名", sources=["recipes"], top_k=2)

    assert result.metadata_expression == 'dish_name LIKE "%%折扣菜%"'
    assert [document.metadata["dish_name"] for document in result.documents] == ["%折扣菜"]
    assert "折扣菜\n\n这条不应该" not in result.context


@pytest.mark.asyncio
async def test_cache_hit_is_checked_against_current_metadata_filter() -> None:
    service = build_test_rag_service()
    service.llm_client = SameRewriteFilterLLM()
    service.generation.llm_client = service.llm_client
    service.metadata_filter.llm_client = service.llm_client
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="meat-doc",
                dish_name="红烧肉",
                category="荤菜",
                difficulty="普通",
                content="# 红烧肉\n\n晚餐推荐晚餐推荐晚餐推荐。",
                source="meat_dish/红烧肉.md",
            ),
            ParsedDocument(
                doc_id="soup-doc",
                dish_name="紫菜蛋花汤",
                category="汤品",
                difficulty="简单",
                content="# 紫菜蛋花汤\n\n晚餐汤。",
                source="soup/紫菜蛋花汤.md",
            ),
        ]
    )

    first = await service.retrieve("推荐晚餐", sources=["recipes"], top_k=1)
    second = await service.retrieve("推荐简单的汤", sources=["recipes"], top_k=1)

    assert first.documents[0].metadata["dish_name"] == "红烧肉"
    assert second.metadata_expression == 'category == "汤品" and difficulty == "简单"'
    assert second.documents[0].metadata["dish_name"] == "紫菜蛋花汤"
    assert any(item == "cache_rejected:recipes:filter" for item in second.trace)


class MissingParentHybridSearch:
    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        return None

    async def hybrid_search(
        self,
        source_name: str,
        query: str,
        expr: str | None,
        top_k: int,
        fetch_multiplier: int,
    ):
        return [
            Document(
                page_content="orphan chunk",
                metadata={
                    "dish_name": "孤儿切片",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "vegetable_dish/orphan.md",
                    "data_source": source_name,
                    "source_type": source_name,
                    "user_id": "GLOBAL",
                    "retrieval_score": 0.9,
                },
            )
        ]


@pytest.mark.asyncio
async def test_rag_retrieval_rejects_hits_without_parent_id() -> None:
    service = build_test_rag_service()
    service.retrieval = MissingParentHybridSearch()

    with pytest.raises(RuntimeError, match="parent_id"):
        await service.retrieve("孤儿切片", sources=["recipes"], top_k=1)


class MissingSourceTypeHybridSearch:
    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        return None

    async def hybrid_search(
        self,
        source_name: str,
        query: str,
        expr: str | None,
        top_k: int,
        fetch_multiplier: int,
    ):
        return [
            Document(
                page_content="missing source type chunk",
                metadata={
                    "parent_id": "fresh-missing-source-type-doc",
                    "dish_name": "缺来源类型菜谱",
                    "category": "素菜",
                    "difficulty": "简单",
                    "source": "vegetable_dish/missing-source-type.md",
                    "data_source": source_name,
                    "user_id": "GLOBAL",
                    "retrieval_score": 0.9,
                },
            )
        ]


@pytest.mark.asyncio
async def test_rag_retrieval_rejects_fresh_hits_missing_source_type() -> None:
    service = build_test_rag_service()
    service.retrieval = MissingSourceTypeHybridSearch()
    service.repository.upsert_documents(
        [
            ParsedDocument(
                doc_id="fresh-missing-source-type-doc",
                dish_name="缺来源类型菜谱",
                category="素菜",
                difficulty="简单",
                content="# 缺来源类型菜谱\n\n这条 hit metadata 不完整。",
                source="vegetable_dish/missing-source-type.md",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="source_type"):
        await service.retrieve("缺来源类型菜谱", sources=["recipes"], top_k=1)


class CrossUserPersonalHybridSearch:
    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        return None

    async def hybrid_search(
        self,
        source_name: str,
        query: str,
        expr: str | None,
        top_k: int,
        fetch_multiplier: int,
    ):
        return [
            Document(
                page_content="cross-user chunk",
                metadata={
                    "parent_id": "fresh-personal-u2-doc",
                    "dish_name": "用户二晚餐偏好",
                    "category": "个人饮食",
                    "difficulty": "普通",
                    "source": "personal/u2/晚餐偏好.md",
                    "data_source": "personal",
                    "source_type": "personal",
                    "user_id": "u2",
                    "retrieval_score": 0.9,
                },
            )
        ]


@pytest.mark.asyncio
async def test_rag_retrieval_rejects_fresh_personal_hits_from_other_users() -> None:
    service = build_test_rag_service()
    service.retrieval = CrossUserPersonalHybridSearch()
    service.repository.upsert_documents(
        [
            ParsedDocument(
                doc_id="fresh-personal-u2-doc",
                dish_name="用户二晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 用户二晚餐偏好\n\n只属于 u2 的个人饮食记录。",
                source="personal/u2/晚餐偏好.md",
                data_source="personal",
                source_type="personal",
                user_id="u2",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="user_id"):
        await service.retrieve("用户二晚餐偏好", user_id="u1", sources=["personal"], top_k=1)


class FailingHybridSearch:
    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        return None

    async def hybrid_search(self, source_name: str, query: str, expr: str | None, top_k: int, fetch_multiplier: int):
        raise RuntimeError("Milvus unavailable")


class FailingIndexHybridSearch(FailingHybridSearch):
    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        raise RuntimeError("Milvus index unavailable")


class FailingNonRuntimeHybridSearch(FailingHybridSearch):
    async def hybrid_search(self, source_name: str, query: str, expr: str | None, top_k: int, fetch_multiplier: int):
        raise ConnectionError("Milvus socket closed")


class FailingPersonalHybridSearch:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        await self.delegate.index_documents(source_name, documents)

    async def hybrid_search(self, source_name: str, query: str, expr: str | None, top_k: int, fetch_multiplier: int):
        if source_name == "personal":
            raise RuntimeError("Milvus personal collection unavailable")
        return await self.delegate.hybrid_search(source_name, query, expr, top_k, fetch_multiplier)


@pytest.mark.asyncio
async def test_multi_source_retrieval_degrades_failed_source_without_losing_healthy_source() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="recipe-healthy-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            ),
            ParsedDocument(
                doc_id="personal-failing-doc",
                dish_name="训练日晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 训练日晚餐偏好\n\n健康晚餐偏好高蛋白和少油。",
                source="personal/u1/训练日晚餐偏好.md",
                data_source="personal",
                source_type="personal",
                user_id="u1",
            ),
        ]
    )
    service.retrieval = FailingPersonalHybridSearch(service.retrieval)

    result = await service.retrieve("健康晚餐", user_id="u1", sources=["recipes", "personal"], top_k=2)

    assert "健康晚餐菜谱" in result.context
    assert all(source.data_source == "recipes" for source in result.sources)
    assert "source_failed:personal" in result.trace


@pytest.mark.asyncio
async def test_multi_source_retrieval_degrades_source_with_missing_parent_document() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="recipe-parent-doc",
                dish_name="健康晚餐菜谱",
                category="素菜",
                difficulty="简单",
                content="# 健康晚餐菜谱\n\n健康晚餐可以选择番茄、鸡蛋和青菜。",
                source="vegetable_dish/健康晚餐菜谱.md",
            ),
            ParsedDocument(
                doc_id="personal-missing-parent-doc",
                dish_name="训练日晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 训练日晚餐偏好\n\n健康晚餐偏好高蛋白和少油。",
                source="personal/u1/训练日晚餐偏好.md",
                data_source="personal",
                source_type="personal",
                user_id="u1",
            ),
        ]
    )
    service.repository._documents.pop("personal-missing-parent-doc")

    result = await service.retrieve("健康晚餐", user_id="u1", sources=["recipes", "personal"], top_k=2)

    assert "健康晚餐菜谱" in result.context
    assert all(source.data_source == "recipes" for source in result.sources)
    assert "source_failed:personal" in result.trace


@pytest.mark.asyncio
async def test_rag_indexing_rejects_empty_parent_document_list() -> None:
    service = build_test_rag_service()

    with pytest.raises(ValueError, match="documents are required"):
        await service.index_parsed_documents([])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_rejects_unknown_document_sources_before_indexing() -> None:
    service = build_test_rag_service()
    document = ParsedDocument(
        doc_id="unknown-source-doc",
        dish_name="外部网页菜谱",
        category="外部",
        difficulty="普通",
        content="# 外部网页菜谱\n\n这个来源不属于 recipes 或 personal。",
        source="web/example.md",
        data_source="web",
        source_type="web",
    )

    with pytest.raises(ValueError, match="unknown document source"):
        await service.index_parsed_documents([document])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_rejects_blank_parent_document_fields_before_indexing() -> None:
    service = build_test_rag_service()
    fields = ["doc_id", "dish_name", "category", "difficulty", "content", "source"]

    for field in fields:
        payload = {
            "doc_id": "blank-field-doc",
            "dish_name": "空字段菜谱",
            "category": "素菜",
            "difficulty": "简单",
            "content": "# 空字段菜谱\n\n正文。",
            "source": "vegetable_dish/空字段菜谱.md",
        }
        payload[field] = "   "
        with pytest.raises(ValueError, match=f"{field} is required"):
            await service.index_parsed_documents([ParsedDocument(**payload)])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_rejects_non_string_parent_document_fields_before_indexing() -> None:
    service = build_test_rag_service()
    payload = {
        "doc_id": None,
        "dish_name": "空字段菜谱",
        "category": "素菜",
        "difficulty": "简单",
        "content": "# 空字段菜谱\n\n正文。",
        "source": "vegetable_dish/空字段菜谱.md",
    }

    with pytest.raises(ValueError, match="doc_id must be text"):
        await service.index_parsed_documents([ParsedDocument(**payload)])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_enforces_source_user_scope_before_indexing() -> None:
    service = build_test_rag_service()
    personal_document = ParsedDocument(
        doc_id="unscoped-personal-doc",
        dish_name="训练日晚餐偏好",
        category="个人饮食",
        difficulty="普通",
        content="# 训练日晚餐偏好\n\n偏好高蛋白、低油。",
        source="personal/GLOBAL/训练日晚餐偏好.md",
        data_source="personal",
        source_type="personal",
        user_id="GLOBAL",
    )
    recipe_document = ParsedDocument(
        doc_id="scoped-recipe-doc",
        dish_name="公共菜谱",
        category="素菜",
        difficulty="简单",
        content="# 公共菜谱\n\n公共菜谱必须属于 GLOBAL。",
        source="vegetable_dish/公共菜谱.md",
        data_source="recipes",
        source_type="recipes",
        user_id="u1",
    )

    with pytest.raises(ValueError, match="personal documents require a non-GLOBAL user_id"):
        await service.index_parsed_documents([personal_document])
    with pytest.raises(ValueError, match="recipe documents must use GLOBAL user_id"):
        await service.index_parsed_documents([recipe_document])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_rejects_source_type_that_conflicts_with_data_source() -> None:
    service = build_test_rag_service()
    document = ParsedDocument(
        doc_id="conflicting-source-type-doc",
        dish_name="公共菜谱",
        category="素菜",
        difficulty="简单",
        content="# 公共菜谱\n\nsource_type 不能和 data_source 冲突。",
        source="vegetable_dish/公共菜谱.md",
        data_source="recipes",
        source_type="personal",
    )

    with pytest.raises(ValueError, match="source_type must match data_source"):
        await service.index_parsed_documents([document])

    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_indexing_normalizes_document_source_scope_before_indexing() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(
        [
            ParsedDocument(
                doc_id="whitespace-personal-doc",
                dish_name="训练日晚餐偏好",
                category="个人饮食",
                difficulty="普通",
                content="# 训练日晚餐偏好\n\n偏好高蛋白、低油。",
                source="personal/u1/训练日晚餐偏好.md",
                data_source=" personal ",
                source_type=" personal ",
                user_id=" u1 ",
            )
        ]
    )

    result = await service.retrieve("训练日晚餐偏好", user_id="u1", sources=["personal"], top_k=3)

    assert "训练日晚餐偏好" in result.context
    assert result.sources[0].data_source == "personal"
    stored = service.repository.list_documents("personal")[0]
    assert stored.data_source == "personal"
    assert stored.source_type == "personal"
    assert stored.user_id == "u1"


@pytest.mark.asyncio
async def test_rag_indexing_does_not_publish_parent_documents_when_milvus_indexing_fails() -> None:
    service = build_test_rag_service()
    service.retrieval = FailingIndexHybridSearch()
    document = ParsedDocument(
        doc_id="transient-doc",
        dish_name="临时菜谱",
        category="素菜",
        difficulty="简单",
        content="# 临时菜谱\n\n暂未完成 Milvus 入库。",
        source="vegetable_dish/临时菜谱.md",
    )

    with pytest.raises(RuntimeError, match="Milvus index unavailable"):
        await service.index_parsed_documents([document])

    assert service.repository.list_documents() == []


class TrackingIndexHybridSearch(FailingHybridSearch):
    def __init__(self) -> None:
        self.chunks_by_source: dict[str, list[Document]] = {}
        self.deleted_parent_ids: list[str] = []

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        self.chunks_by_source.setdefault(source_name, [])
        self.chunks_by_source[source_name].extend(documents)

    def delete_documents(self, source_name: str, documents: list[Document]) -> None:
        parent_ids = {str(document.metadata.get("parent_id")) for document in documents}
        self.deleted_parent_ids.extend(sorted(parent_ids))
        self.chunks_by_source[source_name] = [
            document
            for document in self.chunks_by_source.get(source_name, [])
            if str(document.metadata.get("parent_id")) not in parent_ids
        ]


class FailingUpsertRepository:
    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        raise RuntimeError("PostgreSQL upsert unavailable")


@pytest.mark.asyncio
async def test_rag_indexing_removes_milvus_chunks_when_parent_upsert_fails() -> None:
    service = build_test_rag_service()
    retrieval = TrackingIndexHybridSearch()
    service.retrieval = retrieval
    service.repository = FailingUpsertRepository()
    document = ParsedDocument(
        doc_id="orphan-doc",
        dish_name="临时菜谱",
        category="素菜",
        difficulty="简单",
        content="# 临时菜谱\n\nPostgreSQL 写入失败时不应留下 Milvus chunk。",
        source="vegetable_dish/临时菜谱.md",
    )

    with pytest.raises(RuntimeError, match="PostgreSQL upsert unavailable"):
        await service.index_parsed_documents([document])

    assert retrieval.deleted_parent_ids == ["orphan-doc"]
    assert retrieval.chunks_by_source["recipes"] == []


class FailingSecondSourceIndexHybridSearch(TrackingIndexHybridSearch):
    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        if source_name == "personal":
            raise RuntimeError("Milvus personal index unavailable")
        await super().index_documents(source_name, documents)


@pytest.mark.asyncio
async def test_rag_indexing_removes_prior_source_chunks_when_later_source_indexing_fails() -> None:
    service = build_test_rag_service()
    retrieval = FailingSecondSourceIndexHybridSearch()
    service.retrieval = retrieval
    documents = [
        ParsedDocument(
            doc_id="recipe-before-failure",
            dish_name="公共菜谱",
            category="素菜",
            difficulty="简单",
            content="# 公共菜谱\n\n先写入 recipes collection。",
            source="vegetable_dish/公共菜谱.md",
        ),
        ParsedDocument(
            doc_id="personal-failure",
            dish_name="个人偏好",
            category="个人饮食",
            difficulty="普通",
            content="# 个人偏好\n\npersonal collection 写入失败。",
            source="personal/u1/个人偏好.md",
            data_source="personal",
            source_type="personal",
            user_id="u1",
        ),
    ]

    with pytest.raises(RuntimeError, match="Milvus personal index unavailable"):
        await service.index_parsed_documents(documents)

    assert retrieval.deleted_parent_ids == ["recipe-before-failure"]
    assert retrieval.chunks_by_source["recipes"] == []
    assert service.repository.list_documents() == []


@pytest.mark.asyncio
async def test_rag_retrieval_surfaces_core_milvus_search_failures() -> None:
    service = build_test_rag_service()
    service.retrieval = FailingHybridSearch()

    with pytest.raises(RuntimeError, match="Milvus unavailable"):
        await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=3)


@pytest.mark.asyncio
async def test_rag_retrieval_wraps_non_runtime_core_search_failures() -> None:
    service = build_test_rag_service()
    service.retrieval = FailingNonRuntimeHybridSearch()

    with pytest.raises(RuntimeError, match="source retrieval failed: recipes") as exc_info:
        await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=3)
    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_parent_dedup_prefers_valid_score_over_non_finite_score() -> None:
    service = build_test_rag_service()
    documents = [
        Document(
            page_content="nan chunk",
            metadata={
                "parent_id": "doc-1",
                "dish_name": "番茄炒蛋",
                "source": "vegetable_dish/番茄炒蛋.md",
                "retrieval_score": float("nan"),
            },
        ),
        Document(
            page_content="valid chunk",
            metadata={
                "parent_id": "doc-1",
                "dish_name": "番茄炒蛋",
                "source": "vegetable_dish/番茄炒蛋.md",
                "retrieval_score": 0.8,
            },
        ),
    ]

    deduped = service._dedupe_by_parent(documents)

    assert [document.page_content for document in deduped] == ["valid chunk"]
    assert service._sources(deduped)[0].score == 0.8


def test_cached_document_filter_matches_like_expression() -> None:
    service = build_test_rag_service()
    docs = [
        Document(page_content="", metadata={"dish_name": "番茄炒蛋"}),
        Document(page_content="", metadata={"dish_name": "红烧肉"}),
    ]

    filtered = service._filter_cached_documents(docs, 'dish_name LIKE "%番茄炒蛋%"')

    assert [doc.metadata["dish_name"] for doc in filtered] == ["番茄炒蛋"]


def test_cached_document_filter_keeps_and_inside_string_values() -> None:
    service = build_test_rag_service()
    docs = [
        Document(page_content="", metadata={"dish_name": "salt and pepper chicken", "user_id": "u1"}),
        Document(page_content="", metadata={"dish_name": "salted fish", "user_id": "u1"}),
    ]

    filtered = service._filter_cached_documents(
        docs, 'dish_name == "salt and pepper chicken" and user_id == "u1"'
    )

    assert [doc.metadata["dish_name"] for doc in filtered] == ["salt and pepper chicken"]


def test_cached_document_filter_keeps_equals_inside_like_string() -> None:
    service = build_test_rag_service()
    docs = [
        Document(page_content="", metadata={"dish_name": "酱汁 == 版鸡胸"}),
        Document(page_content="", metadata={"dish_name": "普通鸡胸"}),
    ]

    filtered = service._filter_cached_documents(docs, 'dish_name LIKE "%酱汁 == 版鸡胸%"')

    assert [doc.metadata["dish_name"] for doc in filtered] == ["酱汁 == 版鸡胸"]


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_unknown_sources() -> None:
    service = build_test_rag_service()

    with pytest.raises(ValueError, match="unknown source"):
        await service.retrieve("番茄炒蛋怎么做", sources=["recipes", "web"], top_k=3)


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_blank_query() -> None:
    service = build_test_rag_service()

    with pytest.raises(ValueError, match="query is required"):
        await service.retrieve("   ", sources=["recipes"], top_k=3)


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_non_string_query_before_llm() -> None:
    service = build_test_rag_service()
    no_call_llm = NoCallLLM()
    service.llm_client = no_call_llm
    service.generation.llm_client = no_call_llm
    service.metadata_filter.llm_client = no_call_llm

    with pytest.raises(ValueError, match="query must be text"):
        await service.retrieve(123, sources=["recipes"], top_k=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_non_string_user_scope_before_llm() -> None:
    service = build_test_rag_service()
    no_call_llm = NoCallLLM()
    service.llm_client = no_call_llm
    service.generation.llm_client = no_call_llm
    service.metadata_filter.llm_client = no_call_llm

    with pytest.raises(ValueError, match="user_id must be text"):
        await service.retrieve("训练日晚餐偏好", user_id=123, sources=["personal"], top_k=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_invalid_source_selection_shape_before_llm() -> None:
    service = build_test_rag_service()
    no_call_llm = NoCallLLM()
    service.llm_client = no_call_llm
    service.generation.llm_client = no_call_llm
    service.metadata_filter.llm_client = no_call_llm

    with pytest.raises(ValueError, match="sources must be a list"):
        await service.retrieve("番茄炒蛋怎么做", sources="recipes", top_k=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="source selection must be text"):
        await service.retrieve("番茄炒蛋怎么做", sources=["recipes", 123], top_k=3)  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_non_positive_top_k() -> None:
    service = build_test_rag_service()

    for top_k in (0, -1):
        with pytest.raises(ValueError, match="top_k must be positive"):
            await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=top_k)


@pytest.mark.asyncio
async def test_rag_retrieve_rejects_non_integer_top_k() -> None:
    service = build_test_rag_service()

    for top_k in (True, 1.5, "3"):
        with pytest.raises(ValueError, match="top_k must be positive"):
            await service.retrieve("番茄炒蛋怎么做", sources=["recipes"], top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rag_retrieve_normalizes_source_names() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    result = await service.retrieve("番茄炒蛋怎么做", sources=[" recipes ", "   "], top_k=3)

    assert result.sources[0].data_source == "recipes"
    assert any(item.startswith("hybrid:recipes") for item in result.trace)
    assert not any(item.startswith("hybrid:personal") for item in result.trace)


@pytest.mark.asyncio
async def test_rag_retrieve_deduplicates_source_names_before_retrieval() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    result = await service.retrieve("番茄炒蛋怎么做", sources=[" recipes ", "recipes"], top_k=3)

    assert sum(1 for item in result.trace if item.startswith("hybrid:recipes")) == 1
    assert [source.data_source for source in result.sources] == ["recipes"]


@pytest.mark.asyncio
async def test_rag_retrieve_empty_sources_does_not_default_to_all_sources() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())

    result = await service.retrieve("番茄炒蛋怎么做", sources=[], top_k=3)

    assert result.documents == []
    assert result.sources == []
    assert not any(item.startswith("hybrid:") for item in result.trace)


class NoCallLLM(FakeLLMClient):
    async def complete_text(self, *args, **kwargs) -> str:
        raise AssertionError("empty source retrieval should not call query rewriting LLM")

    async def complete_json(self, *args, **kwargs) -> dict:
        raise AssertionError("empty source retrieval should not call metadata LLM")


@pytest.mark.asyncio
async def test_rag_retrieve_empty_sources_skips_llm_rewrite_and_metadata_filter() -> None:
    service = build_test_rag_service()
    no_call_llm = NoCallLLM()
    service.llm_client = no_call_llm
    service.generation.llm_client = no_call_llm
    service.metadata_filter.llm_client = no_call_llm

    result = await service.retrieve("番茄炒蛋怎么做", sources=[], top_k=3)

    assert result.context == ""
    assert result.documents == []
    assert result.sources == []
    assert result.rewritten_query == "番茄炒蛋怎么做"
    assert result.metadata_expression is None
    assert result.trace == ["sources_empty"]
