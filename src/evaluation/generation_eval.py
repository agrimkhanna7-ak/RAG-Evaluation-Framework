import json
from pathlib import Path

from src.evaluation.dataset import load_evaluation_dataset
from src.retrieval.vector_retriever import retrieve_documents
from src.generation.rag_generator import generate_answer


RESULTS_FILE = Path("data/evaluation/generation_results.json")


def evaluate_generation(
    k: int = 5,
    max_questions: int = 5,
):
    """
    Evaluate generated answers against the expected answers.
    """

    dataset = load_evaluation_dataset()

    # Load previously generated results if the file already exists.
    existing_results = []

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            existing_results = json.load(file)

    # Store existing results by question ID.
    existing_ids = {
        result["id"]
        for result in existing_results
    }

    results = existing_results.copy()

    generated_count = 0

    for item in dataset:

        # Skip questions that have already been evaluated.
        if item["id"] in existing_ids:
            continue

        # Stop after generating the requested number of new answers.
        if generated_count >= max_questions:
            break

        query = item["question"]
        expected_answer = item["expected_answer"]

        retrieved = retrieve_documents(
            query=query,
            k=k,
        )

        retrieved_documents = [
            document
            for document, score in retrieved
        ]

        generated_answer = generate_answer(
            query=query,
            retrieved_documents=retrieved_documents,
        )

        results.append(
            {
                "id": item["id"],
                "question": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
            }
        )

        generated_count += 1

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return results