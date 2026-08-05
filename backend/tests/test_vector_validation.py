import pytest

from app.rag.vector_validation import validate_embedding_vector


def test_validate_embedding_vector_rejects_non_finite_values() -> None:
    with pytest.raises(RuntimeError, match="embedding vector values must be finite numbers"):
        validate_embedding_vector([0.1, float("nan"), 0.3], expected_dim=3)


def test_validate_embedding_vector_rejects_boolean_values() -> None:
    with pytest.raises(RuntimeError, match="embedding vector values must be finite numbers"):
        validate_embedding_vector([0.1, True, 0.3], expected_dim=3)
