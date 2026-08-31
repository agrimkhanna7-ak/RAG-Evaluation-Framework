from src.evaluation.generation_eval import evaluate_generation


results = evaluate_generation(k=5)


print("\n" + "=" * 70)
print("GENERATION EVALUATION")
print("=" * 70)


for result in results:

    print(f"\nID: {result['id']}")
    print(f"Question: {result['question']}")

    print("\nExpected answer:")
    print(result["expected_answer"])

    print("\nGenerated answer:")
    print(result["generated_answer"])

    print("\n" + "-" * 70)