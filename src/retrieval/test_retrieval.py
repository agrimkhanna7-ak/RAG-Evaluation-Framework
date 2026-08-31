from src.retrieval.vector_retriever import retrieve_documents


query = "What are the four functions of the AI Risk Management Framework?"

results = retrieve_documents(
    query=query,
    k=5,
)

print(f"Query: {query}")
print(f"\nRetrieved documents: {len(results)}")

print("\n" + "=" * 80)

for index, (document, score) in enumerate(results, start=1):

    print(f"\nResult {index}")

    print(f"Score: {score}")

    print(
        f"Source: {document.metadata.get('source')}"
    )

    print(
        f"Page: {document.metadata.get('page')}"
    )

    print(
        f"\nContent:\n{document.page_content[:1000]}"
    )

    print("\n" + "-" * 80)