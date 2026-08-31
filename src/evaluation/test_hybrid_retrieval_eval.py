from src.evaluation.hybrid_retrieval_eval import (
    evaluate_hybrid_retrieval,
)


results = evaluate_hybrid_retrieval(
    k=5
)


print("=" * 70)
print("HYBRID RETRIEVAL EVALUATION")
print("=" * 70)


for result in results:

    print(
        f"\nID: {result['id']}"
    )

    print(
        f"Question: {result['question']}"
    )

    print(
        f"Expected: "
        f"{result['expected_source']} "
        f"Page {result['expected_page']}"
    )

    print(
        f"First exact page rank: "
        f"{result['first_exact_page_rank']}"
    )

    print(
        f"First correct document rank: "
        f"{result['first_correct_document_rank']}"
    )