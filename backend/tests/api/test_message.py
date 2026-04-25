from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient

from api.message_citation_utils import remap_response_citations


@dataclass
class DummySourceReference:
    source_id: int
    document_name: str = ""
    chunk_text: str = ""
    relevance_score: float = 0.0
    folder_path: str | None = None



def test_remap_response_citations_maps_deduplicated_source_indexes():
    response = "Ответ с групповыми ссылками [§1, §5] и [§3]."
    chunks_by_source = {
        101: [(1, 0.9, "chunk 1"), (5, 0.6, "chunk 5")],
        202: [(3, 0.8, "chunk 3")],
    }
    source_references = [
        DummySourceReference(
            source_id=101,
            document_name="doc-a.md",
            chunk_text="chunk 1",
            relevance_score=0.95,
            folder_path=None,
        ),
        DummySourceReference(
            source_id=202,
            document_name="doc-b.md",
            chunk_text="chunk 3",
            relevance_score=0.80,
            folder_path=None,
        ),
    ]

    remapped = remap_response_citations(response, chunks_by_source, source_references)

    assert remapped == "Ответ с групповыми ссылками [§1, §1] и [§2]."


def test_remap_response_citations_keeps_unknown_citations_unchanged():
    response = "Ссылка [§9] не должна ломаться."
    chunks_by_source = {101: [(1, 0.9, "chunk 1")]}
    source_references = [
        DummySourceReference(
            source_id=101,
            document_name="doc-a.md",
            chunk_text="chunk 1",
            relevance_score=0.95,
            folder_path=None,
        )
    ]

    remapped = remap_response_citations(response, chunks_by_source, source_references)

    assert remapped == response

@pytest.mark.asyncio
async def test_send_message_success(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test dialogue"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    mock_rag_response = {"response": "PyTorch is a deep learning framework. [§1] [§2]"}

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_rag_response)
        mock_response.raise_for_status = Mock()

        mock_post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "What is PyTorch?"},
        )

    assert response.status_code == 201
    data = response.json()

    assert "message_id" in data
    assert data["user_message"] == "What is PyTorch?"
    assert (
        data["assistant_response"] == "PyTorch is a deep learning framework. [§1] [§2]"
    )
    assert isinstance(data["sources"], list)
    assert isinstance(data["code_executions"], list)
    assert "created_at" in data


@pytest.mark.asyncio
async def test_send_message_dialogue_not_found(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    response = await client.post(
        "/api/message",
        headers={"Authorization": f"Bearer {token}"},
        json={"dialogue_id": 99999, "message": "Test"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_send_message_unauthorized(client: AsyncClient, create_user):
    response = await client.post(
        "/api/message", json={"dialogue_id": 1, "message": "Test"}
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_send_message_rag_api_unavailable(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    import httpx

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Test"},
        )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_set_feedback_like(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"response": "Answer"})
        mock_response.raise_for_status = Mock()
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        message_response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Test"},
        )

    message_id = message_response.json()["message_id"]

    response = await client.post(
        "/api/message/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"message_id": message_id, "feedback": "like"},
    )

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_set_feedback_dislike(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"response": "Answer"})
        mock_response.raise_for_status = Mock()
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        message_response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Test"},
        )

    message_id = message_response.json()["message_id"]

    response = await client.post(
        "/api/message/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"message_id": message_id, "feedback": "dislike"},
    )

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_set_feedback_message_not_found(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    response = await client.post(
        "/api/message/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"message_id": 99999, "feedback": "like"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_send_message_with_code_execution(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    mock_rag_response = {
        "response": "Here's an example:\n```python\nresult = 2 + 2\n```"
    }

    mock_executor_response = {
        "success": True,
        "stdout": "",
        "stderr": "",
        "result": "4",
        "error": None,
    }

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_rag = AsyncMock()
        mock_rag.json = AsyncMock(return_value=mock_rag_response)
        mock_rag.raise_for_status = Mock()

        mock_executor = AsyncMock()
        mock_executor.json = AsyncMock(return_value=mock_executor_response)
        mock_executor.raise_for_status = Mock()

        mock_post = AsyncMock(side_effect=[mock_rag, mock_executor])
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Show me an example"},
        )

    assert response.status_code == 201
    data = response.json()

    assert "code_executions" in data
    assert len(data["code_executions"]) == 1
    assert data["code_executions"][0]["code"] == "result = 2 + 2"
    assert data["code_executions"][0]["success"] is True
    assert data["code_executions"][0]["result"] == "4"


@pytest.mark.asyncio
async def test_send_message_with_multiple_code_blocks(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    mock_rag_response = {
        "response": "First:\n```python\nresult = 1 + 1\n```\nSecond:\n```py\nresult = 2 * 2\n```"
    }

    mock_executor_response_1 = {
        "success": True,
        "stdout": "",
        "stderr": "",
        "result": "2",
        "error": None,
    }

    mock_executor_response_2 = {
        "success": True,
        "stdout": "",
        "stderr": "",
        "result": "4",
        "error": None,
    }

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_rag = AsyncMock()
        mock_rag.json = AsyncMock(return_value=mock_rag_response)
        mock_rag.raise_for_status = Mock()

        mock_executor_1 = AsyncMock()
        mock_executor_1.json = AsyncMock(return_value=mock_executor_response_1)
        mock_executor_1.raise_for_status = Mock()

        mock_executor_2 = AsyncMock()
        mock_executor_2.json = AsyncMock(return_value=mock_executor_response_2)
        mock_executor_2.raise_for_status = Mock()

        mock_post = AsyncMock(side_effect=[mock_rag, mock_executor_1, mock_executor_2])
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Show me examples"},
        )

    assert response.status_code == 201
    data = response.json()

    assert "code_executions" in data
    assert len(data["code_executions"]) == 2
    assert data["code_executions"][0]["result"] == "2"
    assert data["code_executions"][1]["result"] == "4"


@pytest.mark.asyncio
async def test_send_message_with_code_execution_error(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    mock_rag_response = {"response": "```python\nresult = 1 / 0\n```"}

    mock_executor_response = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "result": None,
        "error": "ZeroDivisionError: division by zero",
    }

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_rag = AsyncMock()
        mock_rag.json = AsyncMock(return_value=mock_rag_response)
        mock_rag.raise_for_status = Mock()

        mock_executor = AsyncMock()
        mock_executor.json = AsyncMock(return_value=mock_executor_response)
        mock_executor.raise_for_status = Mock()

        mock_post = AsyncMock(side_effect=[mock_rag, mock_executor])
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Show me an error"},
        )

    assert response.status_code == 201
    data = response.json()

    assert "code_executions" in data
    assert len(data["code_executions"]) == 1
    assert data["code_executions"][0]["success"] is False
    assert "ZeroDivisionError" in data["code_executions"][0]["error"]


@pytest.mark.asyncio
async def test_send_message_code_executor_unavailable(client: AsyncClient, create_user):
    await create_user("test@example.com", "password123", "testuser")
    auth_response = await client.post(
        "/api/user/auth", json={"email": "test@example.com", "password": "password123"}
    )
    token = auth_response.json()["access_token"]

    dialogue_response = await client.post(
        "/api/dialogue",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test"},
    )
    dialogue_id = dialogue_response.json()["dialogue_id"]

    mock_rag_response = {"response": "```python\nresult = 2 + 2\n```"}

    import httpx

    with patch("api.message.controller.httpx.AsyncClient") as mock_client:
        mock_rag = AsyncMock()
        mock_rag.json = AsyncMock(return_value=mock_rag_response)
        mock_rag.raise_for_status = Mock()

        mock_post = AsyncMock(
            side_effect=[mock_rag, httpx.RequestError("Connection failed")]
        )
        mock_client.return_value.__aenter__.return_value.post = mock_post

        response = await client.post(
            "/api/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"dialogue_id": dialogue_id, "message": "Show me code"},
        )

    assert response.status_code == 201
    data = response.json()

    assert "code_executions" in data
    assert len(data["code_executions"]) == 1
    assert data["code_executions"][0]["success"] is False
    assert "unavailable" in data["code_executions"][0]["error"].lower()
