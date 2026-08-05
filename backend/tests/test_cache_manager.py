import math

import pytest

from app.rag.cache.cache_manager import CacheManager


class NoCallKeywordCache:
    async def get(self, source: str, scope: str, query: str):
        raise AssertionError("invalid cache manager config should fail before L1 lookup")

    async def set(self, source: str, scope: str, query: str, documents, ttl: int) -> None:
        raise AssertionError("invalid cache manager config should fail before L1 write")


class NoCallVectorCache:
    async def get(self, source: str, scope: str, query: str, threshold: float):
        raise AssertionError("invalid cache manager config should fail before L2 lookup")

    async def set(self, source: str, scope: str, query: str, documents, ttl: int) -> None:
        raise AssertionError("invalid cache manager config should fail before L2 write")


@pytest.mark.parametrize("ttl_seconds", [0, -1, True, 1.5, "3600"])
def test_cache_manager_rejects_invalid_ttl_before_backend_use(ttl_seconds) -> None:
    with pytest.raises(ValueError, match="cache ttl_seconds must be a positive integer"):
        CacheManager(
            keyword_backend=NoCallKeywordCache(),
            vector_backend=NoCallVectorCache(),
            ttl_seconds=ttl_seconds,
            l2_threshold=0.92,
        )


@pytest.mark.parametrize(
    "l2_threshold", [0.0, -0.1, 1.01, True, math.nan, math.inf, "0.92"]
)
def test_cache_manager_rejects_invalid_l2_threshold_before_backend_use(
    l2_threshold,
) -> None:
    with pytest.raises(ValueError, match="cache l2_threshold must be between 0 and 1"):
        CacheManager(
            keyword_backend=NoCallKeywordCache(),
            vector_backend=NoCallVectorCache(),
            ttl_seconds=3600,
            l2_threshold=l2_threshold,
        )
