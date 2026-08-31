from src.evaluation.retrieval_eval import evaluate_retrieval


results = evaluate_retrieval(k=5)

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

for result in results:

    print(f"\nID: {result['id']}")
    print(f"Question: {result['question']}")

    print(
        f"Expected: "
        f"{result['expected_source']} "
        f"Page {result['expected_page']}"
    )

    print(
        f"First exact page rank: "
        f"{result['first_relevant_rank']}"
    )

    print(
        f"First correct document rank: "
        f"{result['first_source_rank']}"
    )