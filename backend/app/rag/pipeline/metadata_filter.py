from __future__ import annotations

import json
import re

from app.core.llm import OpenAICompatibleClient
from app.database.document_repository import BaseDocumentRepository
from app.rag.milvus_expr import milvus_string, split_milvus_and_expression


class MetadataFilterExtractor:
    """LLM-driven self-querying filter generator with dictionary validation."""

    allowed_fields = {"category", "dish_name", "difficulty"}

    def __init__(self, repository: BaseDocumentRepository, llm_client: OpenAICompatibleClient) -> None:
        self.repository = repository
        self.llm_client = llm_client

    async def build_filter_expression(
        self, raw_query: str, user_id: str | None = None
    ) -> str | None:
        metadata = self.repository.get_metadata_for_filter(user_id)
        system_prompt = (
            "你是 Milvus 元数据自查询过滤器。根据用户原始查询和合法元数据枚举，"
            "输出 JSON：{\"expr\": string}。没有硬约束时输出 {\"expr\": \"NONE\"}。"
            "只能使用字段 category、dish_name、difficulty；只能使用 ==、LIKE、and；"
            "字段值必须来自枚举；LIKE 只能在完整枚举值两侧添加 %，不能使用子串。不要输出代码围栏。"
        )
        user_prompt = json.dumps(
            {
                "query": raw_query,
                "metadata_dictionary": metadata,
                "examples": [
                    {"query": "简单的川菜怎么做", "expr": "category == \"川菜\" and difficulty == \"简单\""},
                    {"query": "红烧肉", "expr": "dish_name LIKE \"%红烧肉%\""},
                    {"query": "推荐几道菜", "expr": "NONE"},
                ],
            },
            ensure_ascii=False,
        )
        try:
            payload = await self.llm_client.complete_json(
                system_prompt, user_prompt, model="fast", temperature=0.0
            )
        except RuntimeError:
            raise
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        expr = str(payload.get("expr", "NONE")).strip()
        if not expr or expr.upper() == "NONE":
            return None
        return self._validate_expression(expr, metadata)

    @staticmethod
    def combine_with_user_scope(expr: str | None, user_id: str | None) -> str | None:
        if not user_id:
            return expr
        user_expr = f"user_id == {milvus_string(user_id)}"
        return f"({expr}) and {user_expr}" if expr else user_expr

    def _validate_expression(self, expr: str, metadata: dict[str, list[str]]) -> str | None:
        clauses = split_milvus_and_expression(expr)
        validated: list[str] = []
        for clause in clauses:
            match = re.fullmatch(
                r'(category|dish_name|difficulty)\s*(==|LIKE)\s*"((?:\\.|[^"])*)"',
                clause,
            )
            if not match:
                return None
            field, operator, raw_value = match.groups()
            value = self._unescape_milvus_string(raw_value)
            allowed_values = metadata.get(field, [])
            if operator == "LIKE":
                if not (value.startswith("%") and value.endswith("%")):
                    return None
                normalized = value[1:-1]
            else:
                if value.startswith("%") or value.endswith("%"):
                    return None
                normalized = value
            if normalized not in allowed_values:
                return None
            validated.append(f"{field} {operator} {milvus_string(value)}")
        return " and ".join(validated) if validated else None

    @staticmethod
    def _unescape_milvus_string(value: str) -> str:
        return value.replace('\\"', '"').replace("\\\\", "\\")
