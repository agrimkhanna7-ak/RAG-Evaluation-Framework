from src.retrieval.hybrid_retriever import (
    CHUNKS,
    BM25,
    tokenize,
)

from src.retrieval.vector_retriever import (
    retrieve_documents,
)


query = "What are the four functions of the AI Risk Management Framework?"

candidate_k = 20


print("=" * 70)
print("HYBRID RETRIEVAL DEBUG")
print("=" * 70)


# ============================================================
# SEMANTIC RESULTS
# ============================================================

print("\nSEMANTIC TOP 20")
print("-" * 70)

semantic_results = retrieve_documents(
    query=query,
    k=candidate_k,
)

for rank, (document, score) in enumerate(
    semantic_results,
    start=1,
):

    print(
        f"Rank {rank} | "
        f"{document.metadata.get('source')} | "
        f"Page {document.metadata.get('page')} | "
        f"Score {score:.4f}"
    )


# ============================================================
# BM25 RESULTS
# ============================================================

print("\nBM25 TOP 20")
print("-" * 70)

query_tokens = tokenize(query)

bm25_scores = BM25.get_scores(
    query_tokens
)

keyword_ranked = sorted(
    zip(CHUNKS, bm25_scores),
    key=lambda x: x[1],
    reverse=True,
)

for rank, (document, score) in enumerate(
    keyword_ranked[:candidate_k],
    start=1,
):

    print(
        f"Rank {rank} | "
        f"{document.metadata.get('source')} | "
        f"Page {document.metadata.get('page')} | "
        f"Score {score:.4f}"
    )