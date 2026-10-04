import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def create_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return the Gemini LLM.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        google_api_key=api_key,
        #temperature=0,#For our RAG evaluation project, we want the model to be as consistent/deterministic as practical
    )

    return llm