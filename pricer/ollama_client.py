import requests
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from config.logging_config import setup_logging

logger = setup_logging()


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            available_models = [model.get("name") for model in models]

            if self.model not in available_models:
                logger.warning("Ollama is running, but model '%s' was not found.", self.model)
                return False

            logger.info("Ollama connection successful. Model '%s' is available.", self.model)
            return True

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to connect to Ollama.")
            raise ConnectionError(
                "Could not connect to Ollama. Make sure Ollama is running on http://localhost:11434."
            ) from e

        except Exception as e:
            logger.exception("Unexpected error while checking Ollama connection.")
            raise RuntimeError(f"Unexpected Ollama connection error: {e}") from e

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json().get("response", "").strip()

            if not result:
                logger.warning("Ollama returned an empty response.")

            logger.info("Generated response successfully from Ollama.")
            return result

        except requests.exceptions.RequestException as e:
            logger.exception("Ollama generation request failed.")
            raise ConnectionError("Failed to generate response from Ollama.") from e

        except Exception as e:
            logger.exception("Unexpected error during Ollama generation.")
            raise RuntimeError(f"Unexpected Ollama generation error: {e}") from e