from __future__ import annotations

from typing import Dict, List

import pandas as pd


def compute_accuracy(rows: List[dict]) -> float:
    if not rows:
        return 0.0
    return sum(int(r["is_correct"]) for r in rows) / len(rows)


def build_summary(rows: List[dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"overall": {}, "by_answer_type": {}, "by_q_type": {}}

    summary = {
        "overall": {
            "num_samples": int(len(df)),
            "accuracy": float(df["is_correct"].mean()),
        },
        "by_answer_type": {},
        "by_q_type": {},
    }

    for answer_type, part in df.groupby("answer_type"):
        summary["by_answer_type"][str(answer_type)] = {
            "num_samples": int(len(part)),
            "accuracy": float(part["is_correct"].mean()),
        }

    if "q_type" in df.columns:
        for q_type, part in df.groupby("q_type"):
            summary["by_q_type"][str(q_type)] = {
                "num_samples": int(len(part)),
                "accuracy": float(part["is_correct"].mean()),
            }

    return summary