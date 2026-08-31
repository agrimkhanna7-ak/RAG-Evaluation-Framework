from src.retrieval.vector_retriever import retrieve_documents
from src.generation.rag_generator import generate_answer


query = "What are the four functions of the AI Risk Management Framework?"


print("Question:")
print(query)

print("\nRetrieving relevant documents...")

results = retrieve_documents(
    query=query,
    k=5,
)

print(f"Retrieved {len(results)} documents.")

documents = [
    document
    for document, score in results
]


print("\nGenerating answer...")

answer = generate_answer(
    query=query,
    retrieved_documents=documents,
)


print("\n" + "=" * 80)

print("FINAL ANSWER:")
print(answer)

print("=" * 80)