from __future__ import annotations

from collections import OrderedDict
import math
from uuid import uuid4

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.database.document_repository import BaseDocumentRepository
from app.rag.document import Document, ParsedDocument


class DocumentProcessor:
    def __init__(self) -> None:
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "header_1"), ("##", "header_2")],
            strip_headers=False,
        )

    def split_markdown(self, document: ParsedDocument) -> list[Document]:
        chunks = self.splitter.split_text(document.content)
        metadata = {
            "source": document.source,
            "parent_id": document.doc_id,
            "dish_name": document.dish_name,
            "category": document.category,
            "difficulty": document.difficulty,
            "is_dish_index": document.is_dish_index,
            "data_source": document.data_source,
            "user_id": document.user_id,
            "source_type": document.source_type,
        }
        return [
            Document(
                page_content=chunk.page_content,
                metadata={**metadata, **chunk.metadata, "chunk_id": str(uuid4())},
            )
            for chunk in chunks
            if chunk.page_content.strip()
        ]

    def post_process_retrieval(
        self, documents: list[Document], repository: BaseDocumentRepository
    ) -> list[Document]:
        best_by_parent: OrderedDict[str, Document] = OrderedDict()
        missing_parent_sources: list[str] = []
        for document in documents:
            parent_id_value = document.metadata.get("parent_id")
            parent_id = parent_id_value.strip() if isinstance(parent_id_value, str) else ""
            if not parent_id:
                missing_parent_sources.append(
                    str(document.metadata.get("source") or "unknown")
                )
                continue
            current = best_by_parent.get(parent_id)
            if current is None or self._score(document) > self._score(current):
                best_by_parent[parent_id] = document
        if missing_parent_sources:
            raise RuntimeError(
                "retrieval documents missing parent_id: "
                + ", ".join(sorted(missing_parent_sources))
            )

        parents = repository.get_parent_documents(list(best_by_parent.keys()))
        missing_parent_ids = sorted(set(best_by_parent) - set(parents))
        if missing_parent_ids:
            raise RuntimeError(f"parent documents missing: {', '.join(missing_parent_ids)}")

        restored: list[Document] = []
        for parent_id, hit in best_by_parent.items():
            parent = parents[parent_id]
            restored.append(
                Document(
                    page_content=parent.content,
                    metadata={
                        **hit.metadata,
                        "restored_parent": True,
                        "parent_source": parent.source,
                    },
                )
            )
        return sorted(restored, key=self._score, reverse=True)

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
