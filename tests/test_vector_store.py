from unittest.mock import Mock

from config import VectorStoreConfig
from vector_store import VectorStore


def build_vector_store(monkeypatch):
    mock_index = Mock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "doc1",
                "score": 0.9,
                "metadata": {"text": "Text 1", "source_id": "doc1"},
            },
            {
                "id": "doc2",
                "score": 0.8,
                "metadata": {"text": "Text 2", "source_id": "doc2"},
            },
        ]
    }
    mock_stats = Mock()
    mock_stats.total_vector_count = 42
    mock_index.describe_index_stats.return_value = mock_stats

    mock_pc = Mock()
    mock_pc.Index.return_value = mock_index
    mock_pc.inference.embed.return_value = [{"values": [0.1, 0.2, 0.3]}]

    monkeypatch.setattr("vector_store.Pinecone", Mock(return_value=mock_pc))

    config = VectorStoreConfig(
        api_key="pinecone-key",
        index_name="test-index",
        embedding_model="multilingual-e5-large",
        chunk_size=10,
        chunk_overlap=2,
    )
    return VectorStore(config), mock_pc, mock_index


def test_initialization_uses_pinecone_index(monkeypatch):
    vector_store, mock_pc, _ = build_vector_store(monkeypatch)

    mock_pc.Index.assert_called_once_with("test-index")
    assert vector_store.index_name == "test-index"
    assert vector_store.embedding_model == "multilingual-e5-large"


def test_chunk_text_splits_long_text(monkeypatch):
    vector_store, _, _ = build_vector_store(monkeypatch)
    text = " ".join([f"word{i}" for i in range(40)])

    chunks = vector_store.chunk_text(text, chunk_size=5, overlap=1)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)


def test_search_embeds_query_and_formats_matches(monkeypatch):
    vector_store, mock_pc, mock_index = build_vector_store(monkeypatch)

    results = vector_store.search("test query", top_k=2)

    mock_pc.inference.embed.assert_called_once_with(
        model="multilingual-e5-large",
        inputs=["test query"],
        parameters={"input_type": "query", "truncate": "END"},
    )
    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
        include_metadata=True,
    )
    assert results[0] == {
        "id": "doc1",
        "text": "Text 1",
        "metadata": {"text": "Text 1", "source_id": "doc1"},
        "distance": 0.09999999999999998,
    }


def test_get_stats(monkeypatch):
    vector_store, _, _ = build_vector_store(monkeypatch)

    stats = vector_store.get_stats()

    assert stats == {
        "collection_name": "test-index",
        "total_documents": 42,
        "embedding_model": "multilingual-e5-large",
        "persist_directory": "pinecone-cloud",
    }


def test_add_documents_embeds_and_upserts(monkeypatch):
    vector_store, mock_pc, mock_index = build_vector_store(monkeypatch)

    vector_store.add_documents([{"id": "doc1", "text": "Document text"}], batch_size=1)

    mock_pc.inference.embed.assert_called_with(
        model="multilingual-e5-large",
        inputs=["Document text"],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    mock_index.upsert.assert_called_once()
