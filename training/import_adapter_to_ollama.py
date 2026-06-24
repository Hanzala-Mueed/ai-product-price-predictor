import subprocess
from pathlib import Path

from config.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTER_DIR = BASE_DIR / "models" / "adapters" / "product_price_lora"
# MODELFILE_PATH = BASE_DIR / "models" / "ollama" / "Modelfile"
MODELFILE_PATH = ADAPTER_DIR / "Modelfile"
FINETUNED_MODEL_NAME = "product-price-predictor-finetuned"


def validate_adapter() -> bool:
    try:
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"Adapter folder not found: {ADAPTER_DIR}")

        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]

        missing_files = [
            file_name for file_name in required_files
            if not (ADAPTER_DIR / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(f"Missing adapter files: {missing_files}")

        logger.info("LoRA adapter validated successfully.")
        return True

    except Exception as e:
        logger.exception("Adapter validation failed.")
        raise RuntimeError(f"Adapter validation failed: {e}") from e


def write_finetuned_modelfile() -> Path:
    try:
        validate_adapter()

        MODELFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        content = f'''FROM llama3.2:latest
ADAPTER adapter_model.safetensors

SYSTEM """
You are an AI product price prediction assistant.

Estimate the realistic US marketplace price of the product.

Rules:
- Respond only with one price number.
- Do not include a dollar sign.
- Do not include explanation.
- Round to the nearest dollar.
- Use learned product pricing patterns from training data.
"""

PARAMETER temperature 0
PARAMETER top_p 0.1
PARAMETER num_predict 20
'''

        MODELFILE_PATH.write_text(content, encoding="utf-8")

        logger.info("Fine-tuned Ollama Modelfile written successfully.")
        return MODELFILE_PATH

    except Exception as e:
        logger.exception("Failed to write fine-tuned Modelfile.")
        raise RuntimeError(f"Failed to write fine-tuned Modelfile: {e}") from e


def create_finetuned_ollama_model() -> bool:
    try:
        write_finetuned_modelfile()

        # command = [
        #     "ollama",
        #     "create",
        #     FINETUNED_MODEL_NAME,
        #     "-f",
        #     str(MODELFILE_PATH),
        # ]

        # result = subprocess.run(
        #     command,
        #     capture_output=True,
        #     text=True,
        #     check=True,
        # )



        command = [
            "ollama",
            "create",
            FINETUNED_MODEL_NAME,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(ADAPTER_DIR),
        )




        logger.info("Fine-tuned Ollama model created: %s", result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.exception("Ollama fine-tuned model creation failed.")
        raise RuntimeError(
            f"Ollama fine-tuned model creation failed: {e.stderr or e.stdout}"
        ) from e

    except Exception as e:
        logger.exception("Failed to create fine-tuned Ollama model.")
        raise RuntimeError(f"Failed to create fine-tuned Ollama model: {e}") from e


if __name__ == "__main__":
    create_finetuned_ollama_model()