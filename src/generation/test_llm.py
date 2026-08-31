from src.generation.llm import create_llm


llm = create_llm()

response = llm.invoke(
    "What is Retrieval-Augmented Generation? Explain in two sentences."
)

print("Gemini response:")
print(response.content)
