
import torch
import re
from collections import defaultdict




def extract_letter(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    text = text.strip()

    pattern = (
        r"(?:"
        r"(?:the\s+)?answer\s*(?::\s*|is\s+)"
        r"|(?:my\s+)?choice\s*(?::\s*|is\s+)"
        r"|(?:the\s+)?option\s*(?::\s*|is\s+)"
        r")?"
        r"(?:\(\s*([A-D])\s*\)|([A-D]))[.)]?"
    )

    match = re.fullmatch(pattern, text, re.IGNORECASE)

    if match:
        return (match.group(1) or match.group(2)).upper()
    pattern_start = r"^\(?([A-D])\)?(?=$|[\s.:\-)]|(?<=\))[.\s:\-])"

    match = re.match(pattern_start, text, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return None



def mmlu_letter_matching(
    model_outputs: list,
    ground_truth: list,
    subjects: list
) -> dict:

    if not (
        len(model_outputs) == len(ground_truth) == len(subjects)
    ):
        raise ValueError("All input lists must have the same length")
    n = len(model_outputs)
    if any(
        not isinstance(x, str) or x not in ("A", "B", "C", "D")
        for x in ground_truth
    ):
        raise ValueError("ground_truth must contain A, B, C or D")
    if n == 0:
        return {
            "overall_accuracy": torch.tensor(0.0),
            "subject_accuracy": {},
            "valid_response_rate": torch.tensor(0.0),
            "total_correct": torch.tensor(0, dtype=torch.int64),
            "total_questions": torch.tensor(0, dtype=torch.int64)
        }

    letter_to_idx = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    }
    extracted = [
        extract_letter(output)
        for output in model_outputs
    ]
    predictions = torch.tensor(
        [
            letter_to_idx[x] if x is not None else -1
            for x in extracted
        ],
        dtype=torch.int64
    )
    targets = torch.tensor(
        [letter_to_idx[x] for x in ground_truth],
        dtype=torch.int64
    )
    valid_mask = predictions != -1
    correct_mask = (predictions == targets) & valid_mask
    overall_accuracy = correct_mask.float().mean()
    valid_response_rate = valid_mask.float().mean()
    total_correct = correct_mask.sum()
    total_questions = torch.tensor(n, dtype=torch.int64)
    subject_indices = defaultdict(list)

    for i, subject in enumerate(subjects):
        subject_indices[subject].append(i)

    subject_accuracy = {}
    for subject, indices in subject_indices.items():
        subject_correct = correct_mask[indices]
        accuracy = subject_correct.float().mean()
        subject_accuracy[subject] = accuracy.item()

    return {
        "overall_accuracy": overall_accuracy,
        "subject_accuracy": subject_accuracy,
        "valid_response_rate": valid_response_rate,
        "total_correct": total_correct,
        "total_questions": total_questions
    }
model_outputs = ["A", "B", "Invalid", "D", "C"]
ground_truth = ["A", "C", "B", "D", "A"]
subjects = ["math", "math", "math", "history", "history"]
results = mmlu_letter_matching(
    model_outputs,
    ground_truth,
    subjects
)
for key, value in results.items():
    print(f"{key}: {value}")
