import json
from pathlib import Path


EVALUATION_FILE = Path(
    "data/evaluation/questions.json"
)


def load_evaluation_dataset():
    """
    Load the evaluation questions and ground-truth answers.
    """

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    return dataset