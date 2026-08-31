from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.simple_chunker import chunk_documents
from src.embeddings.embedding_model import EMBEDDING_MODEL
from src.vectorstore.chroma_store import create_vectorstore


print("Loading PDFs...")

documents = load_all_pdfs()

print(f"Pages loaded: {len(documents)}")

print("\nCreating chunks...")

chunks = chunk_documents(documents)

print(f"Chunks created: {len(chunks)}")

print("\nCreating Chroma vector store...")

vectorstore = create_vectorstore(
    documents=chunks,
    embedding_model_name=EMBEDDING_MODEL,
)

print("\nVector store created successfully!")
print(f"Stored chunks: {vectorstore._collection.count()}")