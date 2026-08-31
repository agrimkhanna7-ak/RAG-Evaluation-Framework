from src.retrieval.hybrid_retriever import hybrid_search


query = "What are the four functions of the AI Risk Management Framework?"


print("=" * 70)
print("HYBRID SEARCH TEST")
print("=" * 70)

print(f"\nQuestion: {query}\n")

results = hybrid_search(
    query=query,
    k=5,
)

for rank, (document, score) in enumerate(
    results,
    start=1,
):

    print(f"Rank {rank}")
    print(
        f"Source: {document.metadata.get('source')}"
    )
    print(
        f"Page: {document.metadata.get('page')}"
    )
    print(
        f"RRF Score: {score:.6f}"
    )
    print(
        f"Content: {document.page_content[:300]}"
    )
    print("-" * 70)