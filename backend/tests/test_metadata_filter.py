import asyncio

import pytest

from app.rag.pipeline.metadata_filter import MetadataFilterExtractor


def test_like_filter_requires_complete_dictionary_value() -> None:
    extractor = MetadataFilterExtractor(repository=None, llm_client=None)
    metadata = {
        "dish_name": ["红烧肉", "番茄炒蛋"],
        "category": ["荤菜", "素菜"],
        "difficulty": ["简单", "普通"],
    }

    assert (
        extractor._validate_expression('dish_name LIKE "%红烧肉%"', metadata)
        == 'dish_name LIKE "%红烧肉%"'
    )
    assert extractor._validate_expression('dish_name LIKE "%肉%"', metadata) is None


def test_user_scope_escapes_milvus_string_literal() -> None:
    expr = MetadataFilterExtractor.combine_with_user_scope(None, 'user"\\id')

    assert expr == 'user_id == "user\\"\\\\id"'


def test_filter_validation_accepts_escaped_dictionary_values() -> None:
    extractor = MetadataFilterExtractor(repository=None, llm_client=None)
    metadata = {
        "dish_name": ['低脂"鸡胸', "黑椒\\牛肉"],
        "category": ["个人知识"],
        "difficulty": ["普通"],
    }

    assert (
        extractor._validate_expression('dish_name == "低脂\\"鸡胸"', metadata)
        == 'dish_name == "低脂\\"鸡胸"'
    )
    assert (
        extractor._validate_expression('dish_name LIKE "%黑椒\\\\牛肉%"', metadata)
        == 'dish_name LIKE "%黑椒\\\\牛肉%"'
    )


def test_filter_validation_keeps_and_inside_string_values() -> None:
    extractor = MetadataFilterExtractor(repository=None, llm_client=None)
    metadata = {
        "dish_name": ["salt and pepper chicken"],
        "category": ["个人知识"],
        "difficulty": ["普通"],
    }

    assert (
        extractor._validate_expression(
            'dish_name == "salt and pepper chicken" and category == "个人知识"',
            metadata,
        )
        == 'dish_name == "salt and pepper chicken" and category == "个人知识"'
    )


def test_filter_validation_rejects_expression_when_any_clause_is_invalid() -> None:
    extractor = MetadataFilterExtractor(repository=None, llm_client=None)
    metadata = {
        "dish_name": ["红烧肉"],
        "category": ["荤菜"],
        "difficulty": ["普通"],
    }

    assert (
        extractor._validate_expression('dish_name == "红烧肉" and calories == "低"', metadata)
        is None
    )
    assert (
        extractor._validate_expression('category == "荤菜" and difficulty == "不存在"', metadata)
        is None
    )


def test_filter_validation_enforces_operator_wildcard_rules() -> None:
    extractor = MetadataFilterExtractor(repository=None, llm_client=None)
    metadata = {
        "dish_name": ["红烧肉"],
        "category": ["荤菜"],
        "difficulty": ["普通"],
    }

    assert extractor._validate_expression('dish_name LIKE "红烧肉"', metadata) is None
    assert extractor._validate_expression('dish_name LIKE "%红烧肉"', metadata) is None
    assert extractor._validate_expression('dish_name LIKE "红烧肉%"', metadata) is None
    assert extractor._validate_expression('dish_name == "%红烧肉%"', metadata) is None


class MetadataRepository:
    def get_metadata_for_filter(self, user_id: str | None = None) -> dict[str, list[str]]:
        return {
            "dish_name": ["番茄炒蛋"],
            "category": ["素菜"],
            "difficulty": ["简单"],
        }


class MalformedMetadataLLM:
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        raise ValueError("LLM response does not contain a JSON object")


class RuntimeMetadataLLM:
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        raise RuntimeError("LLM_API request failed: network down")


class NonObjectMetadataLLM:
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> list[str]:
        return ["expr", "dish_name == \"番茄炒蛋\""]


async def _build_filter_with(llm_client) -> str | None:
    extractor = MetadataFilterExtractor(repository=MetadataRepository(), llm_client=llm_client)
    return await extractor.build_filter_expression("番茄炒蛋怎么做", user_id="u1")


def test_metadata_filter_degrades_to_none_when_llm_returns_malformed_json() -> None:
    assert asyncio.run(_build_filter_with(MalformedMetadataLLM())) is None


def test_metadata_filter_degrades_to_none_when_llm_returns_non_object_payload() -> None:
    assert asyncio.run(_build_filter_with(NonObjectMetadataLLM())) is None


def test_metadata_filter_surfaces_runtime_llm_failures() -> None:
    with pytest.raises(RuntimeError, match="LLM_API request failed"):
        asyncio.run(_build_filter_with(RuntimeMetadataLLM()))
