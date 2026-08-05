from __future__ import annotations

import math
from typing import Any


def _is_finite_embedding_value(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_embedding_vector(vector: list[float], expected_dim: int) -> list[float]:
    if len(vector) != expected_dim:
        raise RuntimeError(
            f"embedding vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )
    if not all(_is_finite_embedding_value(value) for value in vector):
        raise RuntimeError("embedding vector values must be finite numbers")
    return vector


def validate_embedding_vectors(vectors: list[list[float]], expected_dim: int) -> list[list[float]]:
    for vector in vectors:
        validate_embedding_vector(vector, expected_dim)
    return vectors
