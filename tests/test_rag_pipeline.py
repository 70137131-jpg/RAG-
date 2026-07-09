import asyncio
from unittest.mock import AsyncMock, Mock

import rag_pipeline
from rag_pipeline import RAGPipeline


def build_pipeline(monkeypatch, response_text="Mocked answer [Context 1]"):
    mock_vs = Mock()
    mock_vs.search.return_value = [
        {"id": "doc1", "text": "Context 1", "distance": 0.1},
        {"id": "doc2", "text": "Context 2", "distance": 0.2},
    ]

    mock_models = Mock()
    mock_response = Mock()
    mock_response.text = response_text
    mock_models.generate_content = AsyncMock(return_value=mock_response)

    mock_client = Mock()
    mock_client.aio.models = mock_models

    fake_genai = getattr(rag_pipeline, "genai", Mock())
    fake_types = getattr(rag_pipeline, "types", Mock())
    monkeypatch.setattr(rag_pipeline, "genai", fake_genai, raising=False)
    monkeypatch.setattr(rag_pipeline, "types", fake_types, raising=False)
    monkeypatch.setattr(rag_pipeline, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(fake_genai, "Client", Mock(return_value=mock_client))
    monkeypatch.setattr(
        fake_types,
        "GenerateContentConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    pipeline = RAGPipeline(
        vector_store=mock_vs,
        llm_model="gemini-1.5-flash",
        temperature=0.2,
        max_tokens=123,
    )
    return pipeline, mock_vs, mock_models


def test_initialization_uses_google_genai_client(monkeypatch):
    pipeline, _, _ = build_pipeline(monkeypatch)

    assert pipeline.llm_model == "gemini-1.5-flash"
    assert pipeline.temperature == 0.2
    assert pipeline.max_tokens == 123
    assert hasattr(pipeline, "gemini_client")


def test_retrieve_async_calls_vector_store(monkeypatch):
    pipeline, mock_vs, _ = build_pipeline(monkeypatch)

    results = asyncio.run(pipeline.retrieve_async("test query", top_k=2))

    mock_vs.search.assert_called_once_with("test query", 2)
    assert results[0]["text"] == "Context 1"


def test_generate_async_calls_gemini_with_config(monkeypatch):
    pipeline, _, mock_models = build_pipeline(monkeypatch)

    answer = asyncio.run(pipeline.generate_async("Question?", ["Context 1"]))

    assert answer == "Mocked answer [Context 1]"
    call = mock_models.generate_content.await_args.kwargs
    assert call["model"] == "gemini-1.5-flash"
    assert call["config"]["temperature"] == 0.2
    assert call["config"]["max_output_tokens"] == 123


def test_query_async_returns_metadata(monkeypatch):
    pipeline, _, _ = build_pipeline(monkeypatch)

    result = asyncio.run(pipeline.query_async("test question", top_k=2))

    assert result["answer"] == "Mocked answer [Context 1]"
    assert result["metadata"]["num_contexts"] == 2
    assert result["metadata"]["model"] == "gemini-1.5-flash"


def test_greeting_short_circuits_without_retrieval(monkeypatch):
    pipeline, mock_vs, _ = build_pipeline(monkeypatch)

    result = asyncio.run(pipeline.query_async("hello"))

    assert result["metadata"]["retrieved_docs"] == []
    mock_vs.search.assert_not_called()


def test_generation_warns_when_citations_are_missing(monkeypatch):
    pipeline, _, _ = build_pipeline(monkeypatch, response_text="Ungrounded answer")

    answer = asyncio.run(pipeline.generate_async("Question?", ["Context 1"]))

    assert "may not be fully grounded" in answer


def test_generation_does_not_warn_for_document_gap_answer(monkeypatch):
    pipeline, _, _ = build_pipeline(
        monkeypatch,
        response_text="The provided documents do not contain information about what chlorophyll is used for.",
    )

    answer = asyncio.run(pipeline.generate_async("What is chlorophyll used for?", ["Context 1"]))

    assert "may not be fully grounded" not in answer


def test_generate_async_calls_openrouter(monkeypatch):
    mock_vs = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter answer [Context 1]"}}]
    }
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)

    monkeypatch.setattr(rag_pipeline.requests, "post", mock_post)

    pipeline = RAGPipeline(
        vector_store=mock_vs,
        llm_model="openai/gpt-oss-20b:free",
        llm_provider="openrouter",
        openrouter_api_key="test-openrouter-key",
        temperature=0.2,
        max_tokens=123,
    )

    answer = asyncio.run(pipeline.generate_async("Question?", ["Context 1"]))

    assert answer == "OpenRouter answer [Context 1]"
    call = mock_post.call_args.kwargs
    assert call["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert call["json"]["model"] == "openai/gpt-oss-20b:free"
    assert call["json"]["temperature"] == 0.2
    assert call["json"]["max_tokens"] == 123
