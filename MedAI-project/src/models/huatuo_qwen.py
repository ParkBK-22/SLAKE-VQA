from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class HuatuoQwenVLM:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True,
        )
        self.device = device
        if device != "auto":
            self.model.to(device)

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "You are a medical visual question answering assistant. "
            "Answer the question with a short answer only. "
            "If the question is yes/no, answer only yes or no. "
            f"Question: {question}"
        )

    @torch.inference_mode()
    def generate_answer(self, image, question: str, max_new_tokens: int = 16) -> str:
        prompt = self.build_prompt(question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        if self.device != "auto":
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()