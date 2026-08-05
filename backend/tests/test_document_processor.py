import pytest

from app.database.document_repository import BaseDocumentRepository
from app.rag.document import Document, ParsedDocument
from app.rag.pipeline.document_processor import DocumentProcessor


class MissingParentRepository(BaseDocumentRepository):
    def create_schema(self) -> None:
        return None

    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        return None

    def list_documents(self, data_source: str | None = None) -> list[ParsedDocument]:
        return []

    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        return {}

    def init_all_metadata_cache(self) -> None:
        return None


def test_post_process_retrieval_surfaces_missing_parent_documents() -> None:
    processor = DocumentProcessor()
    hits = [
        Document(
            page_content="## 操作\n炒熟。",
            metadata={
                "parent_id": "doc-missing",
                "dish_name": "番茄炒蛋",
                "retrieval_score": 0.9,
            },
        )
    ]

    with pytest.raises(RuntimeError, match="parent documents missing: doc-missing"):
        processor.post_process_retrieval(hits, MissingParentRepository())


class FailingParentRepository(MissingParentRepository):
    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        raise AssertionError("invalid parent_id must not query parent repository")


def test_post_process_retrieval_rejects_non_text_parent_id_before_parent_lookup() -> None:
    processor = DocumentProcessor()
    hits = [
        Document(
            page_content="## 操作\n炒熟。",
            metadata={
                "parent_id": 123,
                "source": "bad-source",
                "dish_name": "番茄炒蛋",
                "retrieval_score": 0.9,
            },
        )
    ]

    with pytest.raises(RuntimeError, match="retrieval documents missing parent_id: bad-source"):
        processor.post_process_retrieval(hits, FailingParentRepository())


class InMemoryParentRepository(MissingParentRepository):
    def __init__(self, documents: list[ParsedDocument]) -> None:
        self.documents = {document.doc_id: document for document in documents}

    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        return {
            parent_id: self.documents[parent_id]
            for parent_id in parent_ids
            if parent_id in self.documents
        }


def test_post_process_retrieval_ignores_malformed_scores_when_restoring_parents() -> None:
    processor = DocumentProcessor()
    repository = InMemoryParentRepository(
        [
            ParsedDocument(
                doc_id="doc-bad-score",
                dish_name="异常分数菜谱",
                category="素菜",
                difficulty="简单",
                content="# 异常分数菜谱\n\n这篇父文档仍然可以还原。",
                source="vegetable_dish/异常分数菜谱.md",
            ),
            ParsedDocument(
                doc_id="doc-valid-score",
                dish_name="有效分数菜谱",
                category="素菜",
                difficulty="简单",
                content="# 有效分数菜谱\n\n有效分数应该排在前面。",
                source="vegetable_dish/有效分数菜谱.md",
            ),
        ]
    )
    hits = [
        Document(
            page_content="## 命中片段",
            metadata={
                "parent_id": "doc-bad-score",
                "dish_name": "异常分数菜谱",
                "retrieval_score": "not-a-number",
            },
        ),
        Document(
            page_content="## 命中片段",
            metadata={
                "parent_id": "doc-valid-score",
                "dish_name": "有效分数菜谱",
                "retrieval_score": 0.8,
            },
        ),
    ]

    restored = processor.post_process_retrieval(hits, repository)

    assert [document.metadata["parent_id"] for document in restored] == [
        "doc-valid-score",
        "doc-bad-score",
    ]
    assert restored[0].page_content == "# 有效分数菜谱\n\n有效分数应该排在前面。"
    assert restored[1].metadata["restored_parent"] is True
