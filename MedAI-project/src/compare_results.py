from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd


CONDITIONS = ["original", "black", "lpf", "hpf", "patch_shuffle"]


def load_summary(summary_path: str) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(condition: str, summary: Dict) -> List[Dict]:
    rows: List[Dict] = []

    overall = summary.get("overall", {})
    meta = summary.get("meta", {})

    rows.append(
        {
            "condition": condition,
            "group": "overall",
            "name": "overall",
            "num_samples": overall.get("num_samples", None),
            "accuracy": overall.get("accuracy", None),
            "split": meta.get("split", None),
            "seed": meta.get("seed", None),
            "use_hf": meta.get("use_hf", None),
        }
    )

    for answer_type, values in summary.get("by_answer_type", {}).items():
        rows.append(
            {
                "condition": condition,
                "group": "answer_type",
                "name": answer_type,
                "num_samples": values.get("num_samples", None),
                "accuracy": values.get("accuracy", None),
                "split": meta.get("split", None),
                "seed": meta.get("seed", None),
                "use_hf": meta.get("use_hf", None),
            }
        )

    for q_type, values in summary.get("by_q_type", {}).items():
        rows.append(
            {
                "condition": condition,
                "group": "q_type",
                "name": q_type,
                "num_samples": values.get("num_samples", None),
                "accuracy": values.get("accuracy", None),
                "split": meta.get("split", None),
                "seed": meta.get("seed", None),
                "use_hf": meta.get("use_hf", None),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_out", type=str, default="outputs")
    parser.add_argument("--save_dir", type=str, default="outputs/compare")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_rows: List[Dict] = []
    overall_rows: List[Dict] = []

    for condition in CONDITIONS:
        summary_path = os.path.join(args.base_out, condition, "summary.json")
        if not os.path.exists(summary_path):
            print(f"[WARN] Missing summary: {summary_path}")
            continue

        summary = load_summary(summary_path)
        rows = flatten_summary(condition, summary)
        all_rows.extend(rows)

        overall = summary.get("overall", {})
        overall_rows.append(
            {
                "condition": condition,
                "num_samples": overall.get("num_samples", None),
                "accuracy": overall.get("accuracy", None),
            }
        )

    if not all_rows:
        raise FileNotFoundError("No summary.json files found.")

    df_all = pd.DataFrame(all_rows)
    df_overall = pd.DataFrame(overall_rows)

    overall_pivot = df_overall[["condition", "accuracy"]].copy()
    overall_pivot = overall_pivot.set_index("condition").reindex(CONDITIONS).reset_index()

    answer_type_df = df_all[df_all["group"] == "answer_type"].copy()
    if not answer_type_df.empty:
        answer_type_pivot = answer_type_df.pivot_table(
            index="condition",
            columns="name",
            values="accuracy",
            aggfunc="first",
        ).reindex(CONDITIONS)
        answer_type_pivot = answer_type_pivot.reset_index()
    else:
        answer_type_pivot = pd.DataFrame()

    df_all.to_csv(os.path.join(args.save_dir, "all_results_long.csv"), index=False)
    df_overall.to_csv(os.path.join(args.save_dir, "overall_results.csv"), index=False)
    overall_pivot.to_csv(os.path.join(args.save_dir, "overall_results_pivot.csv"), index=False)

    if not answer_type_pivot.empty:
        answer_type_pivot.to_csv(
            os.path.join(args.save_dir, "answer_type_results_pivot.csv"),
            index=False,
        )

    print("\n=== Overall Accuracy ===")
    print(overall_pivot.to_string(index=False))

    if not answer_type_pivot.empty:
        print("\n=== Accuracy by Answer Type ===")
        print(answer_type_pivot.to_string(index=False))

    print(f"\nSaved comparison files to: {args.save_dir}")


if __name__ == "__main__":
    main()