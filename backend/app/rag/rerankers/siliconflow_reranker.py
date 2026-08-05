from __future__ import annotations

import logging
import math

import httpx

from app.rag.document import Document

logger = logging.getLogger(__name__)


def _required_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("rerank query must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError("rerank query is required")
    return normalized


def _required_model(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("SiliconFlow rerank model must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError("SiliconFlow rerank model is required")
    return normalized


def _score_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("SiliconFlow rerank min_score must be between 0 and 1")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ValueError("SiliconFlow rerank min_score must be between 0 and 1")
    return normalized


def _document_texts(documents: list[Document]) -> list[str]:
    texts: list[str] = []
    for document in documents:
        if not isinstance(document, Document):
            raise ValueError("rerank documents must be Document instances")
        text = document.page_content.strip()
        if not text:
            raise ValueError("rerank document text is required")
        texts.append(text)
    return texts


class SiliconFlowReranker:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        enabled: bool = True,
        min_score: float = 0.05,
    ) -> None:
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("SiliconFlow API key must be text")
        normalized_api_key = api_key.strip() if isinstance(api_key, str) else None
        self.api_key = normalized_api_key or None
        self.model = _required_model(model)
        self.enabled = enabled and self.api_key is not None
        self.min_score = _score_threshold(min_score)

    async def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not self.enabled or not documents:
            return documents
        normalized_query = _required_query(query)
        document_texts = _document_texts(documents)

        payload = {
            "model": self.model,
            "query": normalized_query,
            "documents": document_texts,
            "top_n": len(document_texts),
            "return_documents": False,
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    "https://api.siliconflow.cn/v1/rerank",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
        except Exception:
            logger.exception("SiliconFlow rerank failed; returning original retrieval order")
            return documents

        if not results:
            return documents

        try:
            ranked: list[Document] = []
            seen_indexes: set[int] = set()
            for item in results:
                index = int(item["index"])
                if index in seen_indexes:
                    raise ValueError(f"rerank response repeated index: {index}")
                seen_indexes.add(index)
                score = float(item.get("relevance_score", 0.0))
                if not math.isfinite(score):
                    raise ValueError(f"rerank response score must be finite: {score}")
                if index < 0 or index >= len(documents) or score < self.min_score:
                    continue
                source = documents[index]
                ranked.append(
                    Document(
                        page_content=source.page_content,
                        metadata={**source.metadata, "rerank_score": score},
                    )
                )
        except Exception:
            logger.exception("SiliconFlow rerank response malformed; returning original retrieval order")
            return documents
        return sorted(
            ranked,
            key=lambda document: float(document.metadata.get("rerank_score", 0.0)),
            reverse=True,
        )
