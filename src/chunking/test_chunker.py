from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.simple_chunker import chunk_documents


documents = load_all_pdfs()

print(f"Pages loaded: {len(documents)}")

chunks = chunk_documents(documents)

print(f"Chunks created: {len(chunks)}")

print("\nFirst chunk:")
print("--------------------")

print("Source:", chunks[0].metadata["source"])
print("Page:", chunks[0].metadata["page"])
print("Characters:", len(chunks[0].page_content))

print("\nContent:")
print(chunks[0].page_content[:1000])