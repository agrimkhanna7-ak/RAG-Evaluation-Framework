from src.embeddings.embedding_model import create_embedding_model


embedding_model = create_embedding_model()

text = "Artificial intelligence risk management"

vector = embedding_model.encode(text)

print("Embedding created successfully.")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])