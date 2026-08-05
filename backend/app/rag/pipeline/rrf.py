from __future__ import annotations

from collections import defaultdict
import math

from app.rag.document import Document


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, ranked_lists: list[list[Document]], top_k: int) -> list[Document]:
        scores: dict[str, float] = defaultdict(float)
        best_docs: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, document in enumerate(ranked, start=1):
                key = str(document.metadata.get("parent_id") or document.metadata.get("chunk_id"))
                scores[key] += 1.0 / (self.k + rank)
                current = best_docs.get(key)
                if current is None or self._score(document) > self._score(current):
                    best_docs[key] = document

        fused = []
        for key, score in scores.items():
            doc = best_docs[key]
            fused.append(Document(doc.page_content, {**doc.metadata, "rrf_score": score}))
        return sorted(fused, key=lambda document: document.metadata["rrf_score"], reverse=True)[:top_k]

    @staticmethod
    def _score(document: Document) -> float:
        value = (
            document.metadata.get("rerank_score")
            or document.metadata.get("retrieval_score")
            or document.metadata.get("dense_score")
            or document.metadata.get("sparse_score")
        )
        if value is None or isinstance(value, bool):
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return score if math.isfinite(score) else 0.0
