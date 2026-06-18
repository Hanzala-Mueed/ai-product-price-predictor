from training.export_to_ollama import (
    MODELFILE_PATH,
    OLLAMA_MODEL_NAME,
    validate_modelfile,
)


def test_ollama_model_name():
    assert OLLAMA_MODEL_NAME == "product-price-predictor"


def test_modelfile_exists():
    assert MODELFILE_PATH.exists()


def test_modelfile_contains_llama_base():
    content = MODELFILE_PATH.read_text(encoding="utf-8")

    assert "FROM llama3.2:latest" in content
    assert "PARAMETER temperature 0" in content


def test_validate_modelfile():
    assert validate_modelfile() is True