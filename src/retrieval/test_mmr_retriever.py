from src.retrieval.mmr_retriever import (
    retrieve_mmr_documents,
)


query = (
    "What are the four functions of the "
    "AI Risk Management Framework?"
)


print("=" * 70)
print("MMR SEARCH TEST")
print("=" * 70)

print(f"\nQuestion: {query}\n")


results = retrieve_mmr_documents(
    query=query,
    k=5,
    fetch_k=20,
    lambda_mult=0.5,
)


for rank, document in enumerate(
    results,
    start=1,
):

    print(f"Rank {rank}")

    print(
        f"Source: "
        f"{document.metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{document.metadata.get('page')}"
    )

    print(
        f"Content: "
        f"{document.page_content[:300]}"
    )

    print("-" * 70)