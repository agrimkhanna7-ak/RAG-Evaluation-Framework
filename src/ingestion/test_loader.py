from pdf_loader import load_all_pdfs


documents = load_all_pdfs()

print(f"Total pages loaded: {len(documents)}")

print("\nFirst document:")
print("Source:", documents[0].metadata["source"])
print("Page:", documents[0].metadata["page"])

print("\nFirst 500 characters:")
print(documents[0].page_content[:500])