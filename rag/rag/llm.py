from __future__ import annotations

import re
from collections.abc import Iterator

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
        timeout: float = 30.0,
        short_timeout: float = 8.0,
    ) -> None:
        self.client = OpenAI(base_url=api_url, api_key=api_key, timeout=timeout)
        self.short_client = OpenAI(
            base_url=api_url, api_key=api_key, timeout=short_timeout
        )
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

    def invoke_once(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming single completion path for short requests."""
        response = self.short_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            stream=False,
        )
        raw = response.choices[0].message.content
        return _clean_response(raw)

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            stream=True,
        )

        for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content
