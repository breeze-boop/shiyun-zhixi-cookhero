from __future__ import annotations

import logging
import math

from app.rag.cache.backends import KeywordCacheBackend, VectorCacheBackend
from app.rag.document import Document

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(
        self,
        keyword_backend: KeywordCacheBackend,
        vector_backend: VectorCacheBackend | None,
        ttl_seconds: int,
        l2_threshold: float,
    ) -> None:
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds <= 0
        ):
            raise ValueError("cache ttl_seconds must be a positive integer")
        if (
            isinstance(l2_threshold, bool)
            or not isinstance(l2_threshold, (int, float))
            or not math.isfinite(float(l2_threshold))
            or not 0.0 < float(l2_threshold) <= 1.0
        ):
            raise ValueError("cache l2_threshold must be between 0 and 1")
        self.keyword_backend = keyword_backend
        self.vector_backend = vector_backend
        self.ttl_seconds = ttl_seconds
        self.l2_threshold = float(l2_threshold)

    async def get(self, source: str, query: str, scope: str) -> tuple[str, list[Document]] | None:
        try:
            l1 = await self.keyword_backend.get(source, scope, query)
            if l1 is not None:
                return "L1", l1
        except Exception:
            logger.exception("L1 retrieval cache lookup failed")

        if not self.vector_backend:
            return None
        try:
            l2 = await self.vector_backend.get(source, scope, query, self.l2_threshold)
            if l2 is not None:
                return "L2", l2
        except Exception:
            logger.exception("L2 retrieval cache lookup failed")
        return None

    async def set(self, source: str, query: str, documents: list[Document], scope: str) -> bool:
        wrote = False
        try:
            await self.keyword_backend.set(source, scope, query, documents, self.ttl_seconds)
            wrote = True
        except Exception:
            logger.exception("L1 retrieval cache write failed")
        if not self.vector_backend:
            return wrote
        try:
            await self.vector_backend.set(source, scope, query, documents, self.ttl_seconds)
            wrote = True
        except Exception:
            logger.exception("L2 retrieval cache write failed")
        return wrote

