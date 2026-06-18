from config.settings import (
    BASE_OLLAMA_MODEL,
    CUSTOM_OLLAMA_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


def test_base_model_name():
    assert BASE_OLLAMA_MODEL == "llama3.2:latest"


def test_custom_model_name():
    assert CUSTOM_OLLAMA_MODEL == "product-price-predictor"


def test_default_model_is_custom_model():
    assert OLLAMA_MODEL == CUSTOM_OLLAMA_MODEL


def test_ollama_base_url():
    assert OLLAMA_BASE_URL == "http://localhost:11434"