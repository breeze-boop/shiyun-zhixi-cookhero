from app.rag.pipeline.rrf import ReciprocalRankFusion
from app.rag.document import Document


def test_rrf_fuses_ranked_dense_and_sparse_lists_by_parent_id() -> None:
    dense = [
        Document("dense-first", {"parent_id": "a", "dense_score": 0.9}),
        Document("dense-second", {"parent_id": "b", "dense_score": 0.8}),
    ]
    sparse = [
        Document("sparse-first", {"parent_id": "b", "sparse_score": 0.95}),
        Document("sparse-second", {"parent_id": "c", "sparse_score": 0.7}),
    ]

    fused = ReciprocalRankFusion(k=60).fuse([dense, sparse], top_k=3)

    assert [doc.metadata["parent_id"] for doc in fused] == ["b", "a", "c"]
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_rrf_ignores_malformed_candidate_scores_when_choosing_parent_document() -> None:
    ranked = [
        Document("malformed", {"parent_id": "a", "dense_score": "high"}),
        Document("valid", {"parent_id": "a", "dense_score": 0.7}),
    ]

    fused = ReciprocalRankFusion(k=60).fuse([ranked], top_k=1)

    assert fused[0].page_content == "valid"
    assert fused[0].metadata["rrf_score"] > 0
