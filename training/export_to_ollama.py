import subprocess
from pathlib import Path

from config.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_MODEL_NAME = "product-price-predictor"
MODELFILE_PATH = BASE_DIR / "models" / "ollama" / "Modelfile"
ADAPTER_DIR = BASE_DIR / "models" / "adapters" / "product_price_lora"


def validate_modelfile() -> bool:
    try:
        if not MODELFILE_PATH.exists():
            raise FileNotFoundError(f"Modelfile not found: {MODELFILE_PATH}")

        content = MODELFILE_PATH.read_text(encoding="utf-8")

        if "FROM llama3.2:latest" not in content:
            raise ValueError("Modelfile must use llama3.2:latest as base model.")

        logger.info("Ollama Modelfile validated successfully.")
        return True

    except Exception as e:
        logger.exception("Failed to validate Ollama Modelfile.")
        raise RuntimeError(f"Failed to validate Ollama Modelfile: {e}") from e


def adapter_exists() -> bool:
    try:
        return ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir())
    except Exception as e:
        logger.exception("Failed to check adapter directory.")
        raise RuntimeError(f"Failed to check adapter directory: {e}") from e


def create_ollama_model() -> bool:
    try:
        validate_modelfile()

        command = [
            "ollama",
            "create",
            OLLAMA_MODEL_NAME,
            "-f",
            str(MODELFILE_PATH),
        ]

        logger.info("Creating Ollama model with command: %s", " ".join(command))

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info("Ollama model created successfully: %s", result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.exception("Ollama model creation failed.")
        raise RuntimeError(
            f"Ollama model creation failed: {e.stderr or e.stdout}"
        ) from e

    except Exception as e:
        logger.exception("Failed to create Ollama model.")
        raise RuntimeError(f"Failed to create Ollama model: {e}") from e


if __name__ == "__main__":
    create_ollama_model()