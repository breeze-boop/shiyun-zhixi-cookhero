from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.llm import OpenAICompatibleClient
from app.database.document_repository import BaseDocumentRepository
from app.rag.cache.cache_manager import CacheManager
from app.rag.document import Document, ParsedDocument, RetrievalResult, RetrievalSource
from app.rag.pipeline.document_processor import DocumentProcessor
from app.rag.pipeline.generation import GenerationIntegrationModule
from app.rag.pipeline.metadata_filter import MetadataFilterExtractor
from app.rag.milvus_expr import split_milvus_and_expression
from app.rag.pipeline.retrieval import MilvusHybridRetriever
from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker

logger = logging.getLogger(__name__)

ALLOWED_RETRIEVAL_SOURCES = {"recipes", "personal"}


@dataclass(slots=True)
class SourceRetrievalBatch:
    source_name: str
    scope: str
    documents: list[Document]
    cache_hit: bool


class RAGService:
    def __init__(
        self,
        settings: Settings,
        repository: BaseDocumentRepository,
        llm_client: OpenAICompatibleClient,
        retrieval: MilvusHybridRetriever,
        cache_manager: CacheManager | None,
        reranker: SiliconFlowReranker,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.llm_client = llm_client
        self.generation = GenerationIntegrationModule(llm_client)
        self.metadata_filter = MetadataFilterExtractor(repository, llm_client)
        self.retrieval = retrieval
        self.processor = DocumentProcessor()
        self.cache_manager = cache_manager
        self.reranker = reranker

    def initialize_storage(self) -> None:
        self.repository.create_schema()
        self.retrieval.create_collections()
        self.repository.init_all_metadata_cache()

    async def index_parsed_documents(self, documents: list[ParsedDocument]) -> None:
        self._normalize_index_documents(documents)
        chunks_by_source: dict[str, list[Document]] = {}
        for document in documents:
            chunks_by_source.setdefault(document.data_source, [])
            chunks_by_source[document.data_source].extend(self.processor.split_markdown(document))
        indexed_chunks_by_source: dict[str, list[Document]] = {}
        try:
            for source_name, chunks in chunks_by_source.items():
                await self.retrieval.index_documents(source_name, chunks)
                indexed_chunks_by_source[source_name] = chunks
        except Exception:
            self._cleanup_indexed_chunks(indexed_chunks_by_source)
            raise
        try:
            self.repository.upsert_documents(documents)
        except Exception:
            self._cleanup_indexed_chunks(indexed_chunks_by_source)
            raise

    @staticmethod
    def _normalize_index_documents(documents: list[ParsedDocument]) -> None:
        if not documents:
            raise ValueError("documents are required")
        normalized_sources: list[str] = []
        for document in documents:
            document.doc_id = RAGService._required_document_text(document.doc_id, "doc_id")
            document.dish_name = RAGService._required_document_text(document.dish_name, "dish_name")
            document.category = RAGService._required_document_text(document.category, "category")
            document.difficulty = RAGService._required_document_text(document.difficulty, "difficulty")
            document.content = RAGService._required_document_text(document.content, "content")
            document.source = RAGService._required_document_text(document.source, "source")
            document.data_source = RAGService._required_document_text(document.data_source, "data_source")
            normalized_sources.append(document.data_source)
        unknown_sources = sorted(set(normalized_sources) - ALLOWED_RETRIEVAL_SOURCES)
        if unknown_sources:
            raise ValueError(f"unknown document source: {', '.join(unknown_sources)}")
        for document in documents:
            data_source = document.data_source
            user_id = RAGService._required_document_text(document.user_id, "user_id")
            source_type = RAGService._optional_document_text(document.source_type, "source_type") or data_source
            if source_type != data_source:
                raise ValueError("source_type must match data_source")
            if data_source == "personal" and user_id == "GLOBAL":
                raise ValueError("personal documents require a non-GLOBAL user_id")
            if data_source == "recipes" and user_id != "GLOBAL":
                raise ValueError("recipe documents must use GLOBAL user_id")
            document.user_id = user_id
            document.source_type = source_type

    @staticmethod
    def _required_document_text(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _optional_document_text(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        return value.strip()

    def _cleanup_indexed_chunks(self, chunks_by_source: dict[str, list[Document]]) -> None:
        for source_name, chunks in chunks_by_source.items():
            try:
                self.retrieval.delete_documents(source_name, chunks)
            except Exception:
                logger.exception("milvus chunk cleanup failed after indexing rollback: %s", source_name)

    @staticmethod
    def _normalize_top_k(top_k: int | None, default_top_k: int) -> int:
        if top_k is None:
            return default_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        return top_k

    @staticmethod
    def _required_retrieval_text(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _optional_retrieval_text(value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_retrieval_sources(sources: list[str] | None) -> list[str]:
        if sources is None:
            return ["recipes", "personal"]
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")
        source_names: list[str] = []
        seen_sources: set[str] = set()
        for item in sources:
            if not isinstance(item, str):
                raise ValueError("source selection must be text")
            source = item.strip()
            if source and source not in seen_sources:
                source_names.append(source)
                seen_sources.add(source)
        unknown_sources = sorted(set(source_names) - ALLOWED_RETRIEVAL_SOURCES)
        if unknown_sources:
            raise ValueError(f"unknown source selection: {', '.join(unknown_sources)}")
        return source_names

    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        sources: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        trace: list[str] = []
        normalized_query = self._required_retrieval_text(query, "query")
        normalized_user_id = self._optional_retrieval_text(user_id, "user_id")
        source_names = self._normalize_retrieval_sources(sources)
        limit = self._normalize_top_k(top_k, self.settings.default_top_k)
        if not source_names:
            trace.append("sources_empty")
            return RetrievalResult(
                query=normalized_query,
                rewritten_query=normalized_query,
                metadata_expression=None,
                context="",
                documents=[],
                sources=[],
                trace=trace,
            )
        rewritten_query = await self.generation.rewrite_query(normalized_query)
        trace.append(f"rewrite:{rewritten_query}")

        metadata_expr = await self.metadata_filter.build_filter_expression(
            normalized_query, normalized_user_id
        )
        trace.append(f"metadata_expr:{metadata_expr or 'NONE'}")

        batches: list[SourceRetrievalBatch] = []
        allow_source_degradation = len(source_names) > 1
        for source_name in source_names:
            try:
                batches.append(
                    await self._retrieve_from_source(
                        source_name=source_name,
                        raw_expr=metadata_expr,
                        rewritten_query=rewritten_query,
                        user_id=normalized_user_id,
                        top_k=limit,
                        trace=trace,
                    )
                )
            except RuntimeError:
                if not allow_source_degradation:
                    raise
                logger.exception("source retrieval failed: %s", source_name)
                trace.append(f"source_failed:{source_name}")
            except Exception as exc:
                if not allow_source_degradation:
                    raise RuntimeError(f"source retrieval failed: {source_name}") from exc
                logger.exception("source retrieval failed: %s", source_name)
                trace.append(f"source_failed:{source_name}")

        cached_documents: list[Document] = []
        fresh_documents: list[Document] = []
        fallback_batches: list[SourceRetrievalBatch] = []
        for batch in batches:
            if not batch.cache_hit:
                fresh_documents.extend(batch.documents)
                continue
            try:
                cached_documents.extend(self._restore_cached_documents(batch.documents))
            except Exception:
                logger.exception("source cache restore failed: %s", batch.source_name)
                trace.append(f"cache_rejected:{batch.source_name}:restore")
                try:
                    fallback_batch = await self._retrieve_from_source(
                        source_name=batch.source_name,
                        raw_expr=metadata_expr,
                        rewritten_query=rewritten_query,
                        user_id=normalized_user_id,
                        top_k=limit,
                        trace=trace,
                        use_cache=False,
                    )
                except RuntimeError:
                    if not allow_source_degradation:
                        raise
                    logger.exception("source retrieval failed after cache restore failure: %s", batch.source_name)
                    trace.append(f"source_failed:{batch.source_name}")
                    continue
                except Exception as exc:
                    if not allow_source_degradation:
                        raise RuntimeError(f"source retrieval failed: {batch.source_name}") from exc
                    logger.exception("source retrieval failed after cache restore failure: %s", batch.source_name)
                    trace.append(f"source_failed:{batch.source_name}")
                    continue
                fallback_batches.append(fallback_batch)
                fresh_documents.extend(fallback_batch.documents)

        reranked = await self._rerank_if_needed(rewritten_query, fresh_documents)
        fresh_scopes = {
            batch.source_name: batch.scope
            for batch in [*batches, *fallback_batches]
            if not batch.cache_hit
        }
        restored_fresh = self._restore_fresh_documents_by_source(
            reranked, allow_source_degradation, trace, fresh_scopes
        )
        await self._write_missed_source_caches([*batches, *fallback_batches], rewritten_query, restored_fresh, trace)
        restored = sorted(cached_documents + restored_fresh, key=self._score, reverse=True)[:limit]
        context = "\n\n---\n\n".join(document.page_content for document in restored)
        return RetrievalResult(
            query=normalized_query,
            rewritten_query=rewritten_query,
            metadata_expression=metadata_expr,
            context=context,
            documents=restored,
            sources=self._sources(restored),
            trace=trace,
        )

    async def generate_answer(self, query: str, result: RetrievalResult) -> str:
        system_prompt = (
            "你是食韵智析的饮食健康智能助手。必须基于检索上下文回答，"
            "如果上下文不足，要明确说明。输出中文，结构清晰，包含可执行建议。"
        )
        user_prompt = f"用户问题：{query}\n\n检索上下文：\n{result.context}"
        return await self.llm_client.complete_text(
            system_prompt, user_prompt, model="reasoning", temperature=0.2
        )

    async def _retrieve_from_source(
        self,
        source_name: str,
        raw_expr: str | None,
        rewritten_query: str,
        user_id: str | None,
        top_k: int,
        trace: list[str],
        use_cache: bool = True,
    ) -> SourceRetrievalBatch:
        if source_name == "personal" and not user_id:
            trace.append("personal_skipped:missing_user_id")
            return SourceRetrievalBatch(source_name, "anonymous", [], cache_hit=False)
        scope = user_id if source_name == "personal" else "global"
        expr = (
            self.metadata_filter.combine_with_user_scope(raw_expr, user_id)
            if source_name == "personal"
            else raw_expr
        )

        if use_cache and self.cache_manager:
            cached = await self.cache_manager.get(source_name, rewritten_query, scope)
            if cached:
                scoped = self._filter_cached_documents_by_scope(
                    cached[1], source_name, scope
                )
                if len(scoped) != len(cached[1]):
                    trace.append(f"cache_rejected:{source_name}:scope")
                filtered = self._filter_cached_documents(scoped, expr)
                if filtered:
                    trace.append(f"cache_hit:{source_name}:{cached[0]}")
                    return SourceRetrievalBatch(source_name, scope, filtered, cache_hit=True)
                if scoped:
                    trace.append(f"cache_rejected:{source_name}:filter")
            trace.append(f"cache_miss:{source_name}")

        hits = await self.retrieval.hybrid_search(
            source_name=source_name,
            query=rewritten_query,
            expr=expr,
            top_k=top_k,
            fetch_multiplier=self.settings.fetch_multiplier,
        )
        deduped = self._dedupe_by_parent(hits)[:top_k]
        trace.append(f"hybrid:{source_name}:{len(deduped)}")
        return SourceRetrievalBatch(source_name, scope, deduped, cache_hit=False)

    def _restore_cached_documents(self, documents: list[Document]) -> list[Document]:
        self._validate_cached_scores(documents)
        if all(document.metadata.get("restored_parent") is True for document in documents):
            self._validate_cached_source_identity(documents)
            return documents
        return self.processor.post_process_retrieval(documents, self.repository)

    @staticmethod
    def _validate_cached_source_identity(documents: list[Document]) -> None:
        for document in documents:
            metadata = document.metadata
            source = metadata.get("source")
            title = metadata.get("dish_name") or source
            if not isinstance(source, str) or not source.strip():
                raise ValueError("cached source must be text")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("cached source title must be text")

    @staticmethod
    def _validate_cached_scores(documents: list[Document]) -> None:
        for document in documents:
            for field in ("rerank_score", "retrieval_score"):
                value = document.metadata.get(field)
                if value is None:
                    continue
                if isinstance(value, bool):
                    raise ValueError(f"cached {field} must be a finite number")
                try:
                    score = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"cached {field} must be a finite number") from exc
                if not math.isfinite(score):
                    raise ValueError(f"cached {field} must be a finite number")

    def _restore_fresh_documents_by_source(
        self,
        documents: list[Document],
        allow_source_degradation: bool,
        trace: list[str],
        fresh_scopes: dict[str, str],
    ) -> list[Document]:
        documents_by_source: dict[str, list[Document]] = {}
        for document in documents:
            source_name = str(document.metadata.get("data_source") or "")
            documents_by_source.setdefault(source_name, []).append(document)

        restored: list[Document] = []
        for source_name, source_documents in documents_by_source.items():
            try:
                self._validate_fresh_documents_source_scope(
                    source_name, source_documents, fresh_scopes.get(source_name, "")
                )
                restored.extend(
                    self.processor.post_process_retrieval(source_documents, self.repository)
                )
            except RuntimeError:
                if not allow_source_degradation:
                    raise
                logger.exception("source parent restore failed: %s", source_name)
                trace.append(f"source_failed:{source_name}")
        return restored

    def _validate_fresh_documents_source_scope(
        self, source_name: str, documents: list[Document], scope: str
    ) -> None:
        if source_name not in ALLOWED_RETRIEVAL_SOURCES:
            raise RuntimeError("retrieval document data_source is invalid")
        if source_name == "recipes" and scope != "global":
            raise RuntimeError("retrieval recipe document scope is invalid")
        if source_name == "personal" and (not scope or scope == "anonymous"):
            raise RuntimeError("retrieval personal document scope is invalid")
        for document in documents:
            metadata = document.metadata
            if str(metadata.get("source_type") or "") != source_name:
                raise RuntimeError("retrieval document source_type is invalid")
            user_id = str(metadata.get("user_id") or "")
            if source_name == "recipes" and user_id != "GLOBAL":
                raise RuntimeError("retrieval recipe document user_id is invalid")
            if source_name == "personal" and user_id != scope:
                raise RuntimeError("retrieval personal document user_id is invalid")

    def _filter_cached_documents_by_scope(
        self, documents: list[Document], source_name: str, scope: str
    ) -> list[Document]:
        return [
            document
            for document in documents
            if self._cached_document_matches_scope(document, source_name, scope)
        ]

    @staticmethod
    def _cached_document_matches_scope(
        document: Document, source_name: str, scope: str
    ) -> bool:
        metadata = document.metadata
        if str(metadata.get("data_source") or "") != source_name:
            return False
        if str(metadata.get("source_type") or "") != source_name:
            return False
        user_id = str(metadata.get("user_id") or "")
        if source_name == "recipes":
            return scope == "global" and user_id == "GLOBAL"
        return bool(scope) and scope != "anonymous" and user_id == scope

    def _filter_cached_documents(self, documents: list[Document], expr: str | None) -> list[Document]:
        if not expr:
            return documents
        return [document for document in documents if self._matches_metadata_expr(document, expr)]

    @staticmethod
    def _matches_metadata_expr(document: Document, expr: str) -> bool:
        metadata = document.metadata
        clauses = split_milvus_and_expression(expr)
        for clause in clauses:
            match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)\s*(==|LIKE)\s*"((?:\\.|[^"])*)"', clause)
            if not match:
                return False
            field, operator, raw_value = match.groups()
            expected = RAGService._unquote_milvus_string(raw_value)
            if operator == "==":
                if str(metadata.get(field)) != expected:
                    return False
            elif operator == "LIKE":
                pattern = (
                    expected[1:-1]
                    if expected.startswith("%") and expected.endswith("%") and len(expected) >= 2
                    else expected
                )
                if pattern not in str(metadata.get(field, "")):
                    return False
        return True

    @staticmethod
    def _unquote_milvus_string(value: str) -> str:
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        return value.replace('\\"', '"').replace('\\\\', '\\')

    async def _write_missed_source_caches(
        self,
        batches: list[SourceRetrievalBatch],
        rewritten_query: str,
        restored_documents: list[Document],
        trace: list[str],
    ) -> None:
        if not self.cache_manager:
            return
        for batch in batches:
            if batch.cache_hit:
                continue
            documents = [
                document
                for document in restored_documents
                if self._cached_document_matches_scope(document, batch.source_name, batch.scope)
            ]
            if not documents:
                continue
            wrote = await self.cache_manager.set(batch.source_name, rewritten_query, documents, batch.scope)
            trace.append(
                f"cache_write:{batch.source_name}"
                if wrote
                else f"cache_write_failed:{batch.source_name}"
            )

    async def _rerank_if_needed(
        self, rewritten_query: str, documents: list[Document]
    ) -> list[Document]:
        return await self.reranker.rerank(rewritten_query, documents)

    @staticmethod
    def _dedupe_by_parent(documents: list[Document]) -> list[Document]:
        best: dict[str, Document] = {}
        for document in documents:
            parent_id = str(document.metadata.get("parent_id", ""))
            if parent_id not in best or RAGService._score(document) > RAGService._score(
                best[parent_id]
            ):
                best[parent_id] = document
        return sorted(best.values(), key=RAGService._score, reverse=True)

    @staticmethod
    def _score(document: Document) -> float:
        value = document.metadata.get("rerank_score")
        if value is None:
            value = document.metadata.get("retrieval_score")
        if value is None or isinstance(value, bool):
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(score):
            return 0.0
        return score

    def _sources(self, documents: list[Document]) -> list[RetrievalSource]:
        sources = []
        for document in documents:
            metadata = document.metadata
            sources.append(
                RetrievalSource(
                    title=str(metadata.get("dish_name") or metadata.get("source")),
                    dish_name=str(metadata.get("dish_name", "")),
                    category=str(metadata.get("category", "")),
                    difficulty=str(metadata.get("difficulty", "")),
                    source=str(metadata.get("source", "")),
                    score=self._score(document),
                    data_source=str(metadata.get("data_source", "recipes")),
                )
            )
        return sources
