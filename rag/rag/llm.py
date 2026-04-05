from __future__ import annotations

import re

from openai import OpenAI


def _clean_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


class ChatModel:
    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ) -> None:
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def invoke(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
        )
        raw = response.choices[0].message.content
        return _clean_response(raw)
