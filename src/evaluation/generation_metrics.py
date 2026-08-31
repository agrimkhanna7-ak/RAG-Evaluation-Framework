import re

from src.embeddings.embedding_model import create_embedding_model


FALLBACK_MESSAGE = (
    "I could not find the answer in the provided documents."
)


def normalize_text(text: str) -> str:
    """
    Normalize text before comparing answers.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    return text.strip()


def exact_match(
    expected_answer: str,
    generated_answer: str,
) -> bool:
    """
    Check whether the expected and generated answers
    are exactly the same after normalization.
    """

    expected = normalize_text(expected_answer)
    generated = normalize_text(generated_answer)

    return expected == generated


def semantic_similarity(
    expected_answer: str,
    generated_answer: str,
) -> float:
    """
    Calculate semantic similarity between the expected
    and generated answers using the local embedding model.
    """

    model = create_embedding_model()

    embeddings = model.encode(
        [
            expected_answer,
            generated_answer,
        ],
        normalize_embeddings=True,
    )

    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    return similarity


def is_answerable(generated_answer: str) -> bool:
    """
    Check whether the RAG system generated an answer
    instead of returning the fallback message.
    """

    return (
        normalize_text(generated_answer)
        != normalize_text(FALLBACK_MESSAGE)
    )