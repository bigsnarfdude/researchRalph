"""
reward.py — correctness checking for GSM8K.
Agents can import and modify reward functions from here.
Read-only interface: check_answer(prediction, ground_truth) -> float
"""
import re


def extract_number(text: str) -> float | None:
    """Extract the final numeric answer from model output."""
    # Look for #### pattern (GSM8K standard)
    match = re.search(r'####\s*([\d,\-\.]+)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def check_answer(prediction: str, ground_truth: str) -> float:
    """
    Binary correctness check.
    Returns 1.0 if correct, 0.0 if wrong.
    """
    pred_num = extract_number(prediction)
    true_num = extract_number(ground_truth)

    if pred_num is None or true_num is None:
        return 0.0

    return 1.0 if abs(pred_num - true_num) < 1e-3 else 0.0


def check_answer_soft(prediction: str, ground_truth: str) -> float:
    """
    Soft reward: 1.0 correct, 0.5 if answer present but wrong format, 0.0 wrong.
    Agents can use this for denser reward signal.
    """
    if check_answer(prediction, ground_truth) == 1.0:
        return 1.0
    # Partial credit: answer token present but wrong
    if extract_number(prediction) is not None:
        return 0.1
    return 0.0
