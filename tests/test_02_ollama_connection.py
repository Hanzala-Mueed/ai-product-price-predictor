import pytest
from pricer.ollama_client import OllamaClient


def test_ollama_client_initializes():
    client = OllamaClient()

    assert client.model == "llama3.2:latest"
    assert client.base_url == "http://localhost:11434"
    assert client.generate_url == "http://localhost:11434/api/generate"


@pytest.mark.integration
def test_ollama_connection():
    client = OllamaClient()

    assert client.check_connection() is True


@pytest.mark.integration
def test_ollama_generate_response():
    client = OllamaClient()

    response = client.generate("Say only: hello")

    assert isinstance(response, str)
    assert len(response) > 0