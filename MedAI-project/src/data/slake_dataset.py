from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class SlakeSample:
    image_id: str
    question_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str
    q_type: str
    lang: str


class SlakeDataset:
    def __init__(self, slake_root: str, split: str = "test", english_only: bool = True):
        self.slake_root = slake_root
        self.split = split
        self.english_only = english_only
        self.samples = self._load_samples()

    def _candidate_annotation_paths(self) -> List[str]:
        return [
            os.path.join(self.slake_root, f"{self.split}.json"),
            os.path.join(self.slake_root, "annotations", f"{self.split}.json"),
            os.path.join(self.slake_root, self.split, f"{self.split}.json"),
            os.path.join(self.slake_root, "imgs", f"{self.split}.json"),
            os.path.join(self.slake_root, "combine", f"{self.split}.json"),
        ]

    def _find_annotation_file(self) -> str:
        for path in self._candidate_annotation_paths():
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not find SLAKE annotation file for split='{self.split}' under {self.slake_root}"
        )

    def _resolve_image_path(self, item: Dict) -> str:
        candidates = []
        if "img_name" in item:
            candidates.append(os.path.join(self.slake_root, item["img_name"]))
            candidates.append(os.path.join(self.slake_root, "imgs", item["img_name"]))
        if "img_path" in item:
            candidates.append(os.path.join(self.slake_root, item["img_path"]))
        if "image" in item and isinstance(item["image"], str):
            candidates.append(os.path.join(self.slake_root, item["image"]))
            candidates.append(os.path.join(self.slake_root, "imgs", item["image"]))
        if "img_id" in item:
            img_id = str(item["img_id"])
            candidates.extend([
                os.path.join(self.slake_root, "imgs", img_id, "source.jpg"),
                os.path.join(self.slake_root, "imgs", img_id, "source.png"),
                os.path.join(self.slake_root, "imgs", f"{img_id}.jpg"),
                os.path.join(self.slake_root, "imgs", f"{img_id}.png"),
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not resolve image path for item: {item}")

    def _infer_answer_type(self, item: Dict) -> str:
        for key in ["answer_type", "q_lang_type", "type"]:
            if key in item and isinstance(item[key], str):
                value = item[key].lower()
                if value in {"open", "closed"}:
                    return value
        answer = str(item.get("answer", "")).strip().lower()
        if answer in {"yes", "no"}:
            return "closed"
        return "open"

    def _load_samples(self) -> List[SlakeSample]:
        ann_path = self._find_annotation_file()
        with open(ann_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: List[SlakeSample] = []
        for idx, item in enumerate(raw):
            lang = str(item.get("q_lang", item.get("lang", "en"))).lower()
            if self.english_only and lang != "en":
                continue

            samples.append(
                SlakeSample(
                    image_id=str(item.get("img_id", item.get("image_id", idx))),
                    question_id=str(item.get("qid", item.get("question_id", idx))),
                    image_path=self._resolve_image_path(item),
                    question=str(item.get("question", "")).strip(),
                    answer=str(item.get("answer", "")).strip(),
                    answer_type=self._infer_answer_type(item),
                    q_type=str(item.get("q_type", "unknown")),
                    lang=lang,
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return {
            "image_id": sample.image_id,
            "question_id": sample.question_id,
            "image": image,
            "image_path": sample.image_path,
            "question": sample.question,
            "answer": sample.answer,
            "answer_type": sample.answer_type,
            "q_type": sample.q_type,
            "lang": sample.lang,
        }