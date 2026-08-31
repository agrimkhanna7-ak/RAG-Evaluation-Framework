import json
from pathlib import Path

from src.evaluation.generation_metrics import (
    exact_match,
    semantic_similarity,
    is_answerable,
)


RESULTS_FILE = Path(
    "data/evaluation/generation_results.json"
)


with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8",
) as file:

    results = json.load(file)


print("=" * 70)
print("GENERATION METRICS")
print("=" * 70)


semantic_scores = []
exact_matches = 0
answerable_count = 0


for result in results:

    expected = result["expected_answer"]
    generated = result["generated_answer"]

    exact = exact_match(
        expected,
        generated,
    )

    similarity = semantic_similarity(
        expected,
        generated,
    )

    answerable = is_answerable(
        generated
    )

    semantic_scores.append(similarity)

    if exact:
        exact_matches += 1

    if answerable:
        answerable_count += 1

    print()
    print(f"ID: {result['id']}")
    print(f"Exact Match: {exact}")
    print(f"Semantic Similarity: {similarity:.4f}")
    print(f"Answerable: {answerable}")


total_questions = len(results)

if total_questions > 0:

    average_similarity = (
        sum(semantic_scores)
        / total_questions
    )

    exact_match_rate = (
        exact_matches
        / total_questions
    )

    answerability_rate = (
        answerable_count
        / total_questions
    )

    fallback_count = (
        total_questions
        - answerable_count
    )

    fallback_rate = (
        fallback_count
        / total_questions
    )

else:

    average_similarity = 0.0
    exact_match_rate = 0.0
    answerability_rate = 0.0
    fallback_count = 0
    fallback_rate = 0.0


print()
print("=" * 70)
print("OVERALL GENERATION METRICS")
print("=" * 70)

print(
    f"Questions evaluated: "
    f"{total_questions}"
)

print(
    f"Average Semantic Similarity: "
    f"{average_similarity:.4f}"
)

print(
    f"Exact Match Rate: "
    f"{exact_match_rate * 100:.2f}%"
)

print(
    f"Answerability Rate: "
    f"{answerability_rate * 100:.2f}%"
)

print(
    f"Fallback Answers: "
    f"{fallback_count}"
)

print(
    f"Fallback Rate: "
    f"{fallback_rate * 100:.2f}%"
)