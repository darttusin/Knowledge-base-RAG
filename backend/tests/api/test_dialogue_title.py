import sys
import types
from types import SimpleNamespace

fake_rag_service_module = types.ModuleType("services.rag_service")
fake_rag_service_module.RagService = object
sys.modules.setdefault("services.rag_service", fake_rag_service_module)

from api.dialogue.controller import generate_dialogue_title


def _make_rag_service(chat_model: object) -> SimpleNamespace:
    return SimpleNamespace(chat_model=chat_model)


def _stream_chunk(content: str = "", finish_reason: str | None = None) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def test_generate_dialogue_title_non_stream_uses_invoke_once_and_normalizes():
    class NonStreamChatModel:
        def invoke_once(self, messages, temperature=None, max_tokens=None):
            assert isinstance(messages, list)
            assert temperature == 0.2
            assert max_tokens == 16
            return 'Title: "  Practical PyTorch Learning Rate Schedulers  "'

    rag_service = _make_rag_service(NonStreamChatModel())
    title = generate_dialogue_title(
        "How to set up a learning rate scheduler in PyTorch?",
        rag_service,
    )

    assert title == "Practical PyTorch Learning Rate Schedulers"


def test_generate_dialogue_title_stream_aggregates_until_finish_reason():
    class StreamChatModel:
        def invoke(self, _messages, temperature=None, max_tokens=None):
            assert temperature == 0.2
            assert max_tokens == 16
            return iter(
                [
                    _stream_chunk('Title: "'),
                    _stream_chunk("Deep Learning Optimization Tricks"),
                    _stream_chunk('"', finish_reason="stop"),
                    _stream_chunk("ignored after stop"),
                ]
            )

    rag_service = _make_rag_service(StreamChatModel())
    title = generate_dialogue_title(
        "Can you explain practical deep learning optimization tricks?",
        rag_service,
    )

    assert title == "Deep Learning Optimization Tricks"
