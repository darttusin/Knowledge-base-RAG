"""Service for generating dialogue titles using LLM."""

import re
from collections.abc import Iterable

from loguru import logger

from services.rag_service import RagService


def generate_dialogue_title(question: str, rag_service: RagService) -> str:
    """Generate a title for a dialogue based on the first question.

    Args:
        question: First user question in the dialogue
        rag_service: RAG service with access to the chat model

    Returns:
        Generated title (3-6 words)
    """
    prompt = f"""Generate a short, concise title (3-6 words) for a conversation that starts with this question.
The title should capture the main topic or intent of the question.
Output ONLY the title, nothing else.

Question: {question}
Title:"""

    def _fallback_title() -> str:
        words = question.split()[:6]
        return " ".join(words) if words else "New conversation"

    def _extract_stream_content(chunk: object) -> tuple[str, bool]:
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if not choices:
                return "", False
            choice = choices[0] or {}
            delta = choice.get("delta", {}) or {}
            content = delta.get("content", "") or ""
            return content, bool(choice.get("finish_reason"))

        choices = getattr(chunk, "choices", None)
        if not choices:
            return "", False
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", "") if delta else ""
        finish_reason = getattr(choice, "finish_reason", None)
        return content or "", bool(finish_reason)

    def _normalize_title(raw_title: str) -> str:
        title = raw_title.strip()
        title = re.sub(r"[\r\n]+", " ", title)
        title = title.strip().strip("\"'`")
        title = re.sub(
            r"^(title|topic|summary|заголовок)\s*[:\-]\s*",
            "",
            title,
            flags=re.IGNORECASE,
        )
        title = title.replace('"', "").replace("'", "").replace("`", "")
        title = re.sub(r"\s+", " ", title).strip()

        if not title:
            return ""

        words = title.split()
        if len(words) > 6:
            words = words[:6]
        if len(words) < 3:
            return ""
        return " ".join(words)

    try:
        messages = [{"role": "user", "content": prompt}]
        chat_model = rag_service.chat_model

        if hasattr(chat_model, "invoke_once"):
            response = chat_model.invoke_once(messages, max_tokens=16, temperature=0.2)
        else:
            response = chat_model.invoke(messages, max_tokens=16, temperature=0.2)

        if isinstance(response, str):
            title = _normalize_title(response)
            return title if title else _fallback_title()

        if isinstance(response, Iterable):
            chunks: list[str] = []
            for chunk in response:
                content, is_finished = _extract_stream_content(chunk)
                if content:
                    chunks.append(content)
                if is_finished:
                    break
            title = _normalize_title("".join(chunks))
            return title if title else _fallback_title()

        return _fallback_title()
    except Exception as e:
        logger.warning(f"Failed to generate dialogue title: {e}")
        return _fallback_title()
