from src.retrieval.vector_retriever import retrieve_documents
from src.generation.rag_generator import generate_answer


query = "What are the four functions of the AI Risk Management Framework?"


print("Question:")
print(query)

print("\nRetrieving relevant documents...")

retrieved = retrieve_documents(
    query=query,
    k=5,
)

documents = [
    document
    for document, score in retrieved
]

print(f"Retrieved {len(documents)} documents.")


print("\nGenerating answer...")

answer = generate_answer(
    query=query,
    retrieved_documents=documents,
)


print("\n" + "=" * 70)
print("GENERATED ANSWER")
print("=" * 70)

print(answer)