from __future__ import annotations

import time
from collections import defaultdict
from threading import RLock

from app.core.config import Settings
from app.database.document_repository import BaseDocumentRepository
from app.database.user_repository import DietPlanRecord, MealCheckinRecord, NutritionReportRecord
from uuid import uuid4
from app.rag.cache.backends import CacheEntry, KeywordCacheBackend, RedisKeywordCache, VectorCacheBackend
from app.rag.cache.cache_manager import CacheManager
from app.rag.document import Document, ParsedDocument
from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker
from app.services.rag_service import RAGService


class FakeLLMClient:
    async def complete_text(
        self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.1
    ) -> str:
        if "查询改写" in system_prompt:
            if "番茄炒蛋怎么做" in user_prompt:
                return "番茄炒蛋的详细制作步骤和做法"
            if "推荐简单的汤" in user_prompt:
                return "推荐几道简单汤品"
            return user_prompt.replace("原始查询：", "")
        return "基于检索上下文生成的饮食建议。"

    async def complete_json(
        self, system_prompt: str, user_prompt: str, *, model: str = "fast", temperature: float = 0.0
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            if "diet_planning_expert" in user_prompt:
                return {"thought": "饮食计划专家调用计划工具", "action": "diet_plan", "action_input": {"goal": "减脂高蛋白", "days": 3, "context_query": "番茄炒蛋"}}
            if "meal_record_expert" in user_prompt:
                return {"thought": "记录专家调用打卡工具", "action": "meal_checkin", "action_input": {"meal_time": "dinner", "description": "番茄炒蛋和米饭"}}
            if "nutrition_analysis_expert" in user_prompt:
                return {"thought": "营养专家调用分析工具", "action": "nutrition_analysis", "action_input": {"date": "2026-07-30"}}
        if "工具调度器" in system_prompt or "工具调度" in system_prompt:
            if '"name": "diet_planning_expert"' in user_prompt and '"name": "diet_plan"' not in user_prompt:
                return {"thought": "用户需要饮食计划专家", "action": "diet_planning_expert", "action_input": {"message": "帮我制定3天减脂高蛋白饮食计划"}}
            if "饮食计划" in user_prompt or "减脂" in user_prompt:
                return {"thought": "用户需要生成饮食计划", "action": "diet_plan", "action_input": {"goal": "减脂高蛋白", "days": 3, "context_query": "番茄炒蛋"}}
            if "打卡" in user_prompt:
                return {"thought": "用户需要记录餐食", "action": "meal_checkin", "action_input": {"meal_time": "dinner", "description": "番茄炒蛋和米饭"}}
            if "营养分析" in user_prompt:
                return {"thought": "用户需要营养分析", "action": "nutrition_analysis", "action_input": {"date": "2026-07-30"}}
            return {"thought": "默认检索知识库", "action": "knowledge_base_search", "action_input": {"query": "番茄炒蛋怎么做"}}
        if "营养分析" in system_prompt or "营养" in system_prompt:
            return {"content": "蛋白质摄入稳定，晚餐油脂需要控制。", "metrics": {"protein": "24g", "carbs": "38g", "fat": "16g", "energy": "620kcal", "risk": "low"}}
        if "番茄炒蛋" in user_prompt:
            return {"expr": "dish_name == \"番茄炒蛋\""}
        if "简单" in user_prompt and "汤" in user_prompt:
            return {"expr": "category == \"汤品\" and difficulty == \"简单\""}
        return {"expr": "NONE"}

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeDocumentRepository(BaseDocumentRepository):
    def __init__(self) -> None:
        self._documents: dict[str, ParsedDocument] = {}
        self._lock = RLock()

    def create_schema(self) -> None:
        return None

    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        with self._lock:
            for document in documents:
                self._documents[document.doc_id] = document
            self.init_all_metadata_cache()

    def list_documents(self, data_source: str | None = None) -> list[ParsedDocument]:
        with self._lock:
            documents = list(self._documents.values())
        if data_source is None:
            return documents
        return [doc for doc in documents if doc.data_source == data_source]

    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        with self._lock:
            return {
                parent_id: self._documents[parent_id]
                for parent_id in parent_ids
                if parent_id in self._documents
            }

    def init_all_metadata_cache(self) -> None:
        with self._lock:
            global_values: dict[str, set[str]] = defaultdict(set)
            user_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for document in self._documents.values():
                bucket = global_values if document.user_id == "GLOBAL" else user_values[document.user_id]
                bucket["category"].add(document.category)
                bucket["dish_name"].add(document.dish_name)
                bucket["difficulty"].add(document.difficulty)
        self.__class__._global_cache = {key: sorted(values) for key, values in global_values.items()}
        self.__class__._user_cache = {
            user_id: {key: sorted(values) for key, values in values_by_key.items()}
            for user_id, values_by_key in user_values.items()
        }


class FakeKeywordCache(KeywordCacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    async def get(self, source: str, scope: str, query: str) -> list[Document] | None:
        key = RedisKeywordCache._key(source, scope, query)
        entry = self._store.get(key)
        if not entry or entry.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return entry.value

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        self._store[RedisKeywordCache._key(source, scope, query)] = CacheEntry(
            value=documents, expires_at=time.time() + ttl
        )


class FakeVectorCache(VectorCacheBackend):
    async def get(
        self, source: str, scope: str, query: str, threshold: float
    ) -> list[Document] | None:
        return None

    async def set(
        self, source: str, scope: str, query: str, documents: list[Document], ttl: int
    ) -> None:
        return None


class FakeHybridRetriever:
    def __init__(self) -> None:
        self._chunks_by_source: dict[str, list[Document]] = {}

    def create_collections(self) -> None:
        return None

    async def index_documents(self, source_name: str, documents: list[Document]) -> None:
        self._chunks_by_source[source_name] = list(documents)

    async def hybrid_search(
        self, source_name: str, query: str, expr: str | None, top_k: int, fetch_multiplier: int
    ) -> list[Document]:
        candidates = [
            document
            for document in self._chunks_by_source.get(source_name, [])
            if self._matches_expr(document, expr)
        ]
        scored = []
        for document in candidates:
            text = " ".join([document.page_content, str(document.metadata.get("dish_name", ""))])
            score = sum(1 for char in set(query) if char in text) / max(len(set(query)), 1)
            scored.append(Document(document.page_content, {**document.metadata, "retrieval_score": score}))
        return sorted(scored, key=lambda item: item.metadata["retrieval_score"], reverse=True)[
            : max(top_k * fetch_multiplier, top_k)
        ]

    @staticmethod
    def _matches_expr(document: Document, expr: str | None) -> bool:
        if not expr:
            return True
        metadata = document.metadata
        clauses = [clause.strip(" ()") for clause in expr.split(" and ")]
        for clause in clauses:
            if "==" not in clause:
                continue
            field, value = clause.split("==", 1)
            if str(metadata.get(field.strip())) != value.strip().strip('"'):
                return False
        return True


class FakeUserRepository:
    def __init__(self) -> None:
        self.plans: list[DietPlanRecord] = []
        self.checkins: list[MealCheckinRecord] = []
        self.reports: list[NutritionReportRecord] = []

    def create_schema(self) -> None:
        return None

    def create_diet_plan(self, user_id: str, goal: str, days: int, content: str) -> DietPlanRecord:
        plan = DietPlanRecord(str(uuid4()), user_id, goal, days, content)
        self.plans.insert(0, plan)
        return plan

    def list_diet_plans(self, user_id: str, limit: int = 5) -> list[DietPlanRecord]:
        return [item for item in self.plans if item.user_id == user_id][:limit]

    def create_meal_checkin(
        self, user_id: str, meal_time: str, description: str, image_analysis: dict
    ) -> MealCheckinRecord:
        checkin = MealCheckinRecord(str(uuid4()), user_id, meal_time, description, image_analysis)
        self.checkins.insert(0, checkin)
        return checkin

    def list_meal_checkins(self, user_id: str, limit: int = 20) -> list[MealCheckinRecord]:
        return [item for item in self.checkins if item.user_id == user_id][:limit]

    def save_nutrition_report(
        self, user_id: str, date: str, content: str, metrics: dict
    ) -> NutritionReportRecord:
        report = NutritionReportRecord(str(uuid4()), user_id, date, content, metrics)
        self.reports.append(report)
        return report


def build_test_rag_service() -> RAGService:
    settings = Settings(
        rerank_enabled=False,
        llm_api_key="test",
        embedding_api_key="test",
        embedding_dim=3,
    )
    return RAGService(
        settings=settings,
        repository=FakeDocumentRepository(),
        llm_client=FakeLLMClient(),
        retrieval=FakeHybridRetriever(),
        cache_manager=CacheManager(
            keyword_backend=FakeKeywordCache(),
            vector_backend=FakeVectorCache(),
            ttl_seconds=settings.cache_ttl_seconds,
            l2_threshold=settings.l2_similarity_threshold,
        ),
        reranker=SiliconFlowReranker(api_key=None, model=settings.siliconflow_rerank_model, enabled=False),
    )
