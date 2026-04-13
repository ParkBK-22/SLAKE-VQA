import re
import string
from typing import Optional


PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_closed_answer(text: str) -> Optional[str]:
    norm = normalize_text(text)
    tokens = norm.split()
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    if norm.startswith("yes"):
        return "yes"
    if norm.startswith("no"):
        return "no"
    return None


def open_match(pred: str, gt: str, mode: str = "exact") -> bool:
    pred_n = normalize_text(pred)
    gt_n = normalize_text(gt)
    if mode == "exact":
        return pred_n == gt_n
    if mode == "substring":
        return gt_n in pred_n
    raise ValueError(f"Unknown open match mode: {mode}")