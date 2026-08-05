from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    dish_name: str
    category: str
    difficulty: str
    content: str
    source: str
    data_source: str = "recipes"
    source_type: str = "recipes"
    user_id: str = "GLOBAL"
    is_dish_index: bool = False


@dataclass(slots=True)
class RetrievalSource:
    title: str
    dish_name: str
    category: str
    difficulty: str
    source: str
    score: float | None = None
    data_source: str = "recipes"


@dataclass(slots=True)
class RetrievalResult:
    query: str
    rewritten_query: str
    metadata_expression: str | None
    context: str
    documents: list[Document]
    sources: list[RetrievalSource]
    trace: list[str]

