from unittest.mock import Mock

from data_loader import SQuADLoader


def test_initialization():
    loader = SQuADLoader(dataset_name="squad_v2", split="validation")

    assert loader.dataset_name == "squad_v2"
    assert loader.split == "validation"
    assert loader.dataset is None


def test_load_with_max_samples(monkeypatch):
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.select = Mock(return_value=mock_dataset)
    mock_load_dataset = Mock(return_value=mock_dataset)
    monkeypatch.setattr("data_loader.load_dataset", mock_load_dataset)

    loader = SQuADLoader(dataset_name="squad_v2", split="validation")
    loader.load(max_samples=50)

    mock_load_dataset.assert_called_once_with("squad_v2", split="validation")
    mock_dataset.select.assert_called_once()
    assert loader.dataset is mock_dataset


def test_get_contexts(monkeypatch):
    mock_data = [
        {"context": "Context 1", "question": "Q1", "answers": {"text": ["A1"], "answer_start": [0]}},
        {"context": "Context 1", "question": "Q2", "answers": {"text": ["A2"], "answer_start": [0]}},
        {"context": "Context 2", "question": "Q3", "answers": {"text": ["A3"], "answer_start": [0]}},
    ]
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=3)
    monkeypatch.setattr("data_loader.load_dataset", Mock(return_value=mock_dataset))

    loader = SQuADLoader(dataset_name="squad_v2", split="validation")
    loader.load()
    contexts = loader.get_contexts()

    assert len(contexts) == 2
    assert all("id" in context and "text" in context for context in contexts)


def test_get_qa_pairs(monkeypatch):
    mock_data = [
        {
            "id": "1",
            "context": "Context 1",
            "question": "Q1",
            "answers": {"text": ["Answer 1"], "answer_start": [0]},
        },
        {
            "id": "2",
            "context": "Context 2",
            "question": "Q2",
            "answers": {"text": [], "answer_start": []},
        },
        {
            "id": "3",
            "context": "Context 3",
            "question": "Q3",
            "answers": {"text": ["Answer 3"], "answer_start": [0]},
        },
    ]
    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
    mock_dataset.__len__ = Mock(return_value=3)
    monkeypatch.setattr("data_loader.load_dataset", Mock(return_value=mock_dataset))

    loader = SQuADLoader(dataset_name="squad_v2", split="validation")
    loader.load()
    qa_pairs = loader.get_qa_pairs()

    assert len(qa_pairs) == 2
    for qa_pair in qa_pairs:
        assert "id" in qa_pair
        assert "question" in qa_pair
        assert "answers" in qa_pair
        assert qa_pair["answers"]
