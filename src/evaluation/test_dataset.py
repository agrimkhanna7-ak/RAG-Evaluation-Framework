from src.evaluation.dataset import load_evaluation_dataset


dataset = load_evaluation_dataset()


print(f"Evaluation questions: {len(dataset)}")


for item in dataset:

    print("\n" + "=" * 60)

    print(f"ID: {item['id']}")
    print(f"Question: {item['question']}")
    print(f"Expected answer: {item['expected_answer']}")
    print(f"Source: {item['source']}")
    print(f"Page: {item['page']}")