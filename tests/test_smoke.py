from unittest.mock import Mock


def test_imports():
    import app
    import config
    import config_utils
    import data_loader
    import rag_pipeline
    import vector_store

    assert app.app is not None
    assert config.RAGConfig.from_env() is not None
    assert callable(config_utils.create_rag_pipeline_from_config)
    assert data_loader.SQuADLoader is not None
    assert rag_pipeline.RAGPipeline is not None
    assert vector_store.VectorStore is not None


def test_config_creation():
    from config import RAGConfig

    config = RAGConfig.from_env()

    assert config.vector_store is not None
    assert config.llm is not None
    assert config.redis is not None
    assert config.data is not None


def test_data_loader_initialization():
    from data_loader import SQuADLoader

    loader = SQuADLoader(dataset_name="squad_v2", split="validation")

    assert loader.dataset_name == "squad_v2"
    assert loader.split == "validation"


def test_config_utils_create_components(monkeypatch):
    import config_utils
    from config import RAGConfig

    mock_vector_store = Mock()
    mock_pipeline = Mock()

    monkeypatch.setattr(config_utils, "VectorStore", Mock(return_value=mock_vector_store))
    monkeypatch.setattr(config_utils, "RAGPipeline", Mock(return_value=mock_pipeline))

    config = RAGConfig.from_env()
    vector_store = config_utils.create_vector_store_from_config(config.vector_store)
    pipeline = config_utils.create_rag_pipeline_from_config(config, vector_store)

    assert vector_store is mock_vector_store
    assert pipeline is mock_pipeline
    config_utils.RAGPipeline.assert_called_once()
    kwargs = config_utils.RAGPipeline.call_args.kwargs
    assert "llm_provider" in kwargs
    assert "openrouter_api_key" in kwargs
