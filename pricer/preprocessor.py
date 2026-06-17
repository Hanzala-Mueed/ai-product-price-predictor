from pricer.ollama_client import OllamaClient
from config.logging_config import setup_logging

logger = setup_logging()

SYSTEM_PROMPT = """You are a product data preprocessing assistant.

Create a concise product summary for price prediction.

Respond only in this exact format:
Title: rewritten short precise title
Category: product category
Brand: brand name or Unknown
Description: one sentence product description
Details: one sentence with the most important features

Rules:
- Do not include part numbers.
- Do not include model numbers.
- Do not include marketing hype.
- Keep the response short and clean.
"""


class ProductPreprocessor:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def build_user_prompt(self, product_text: str) -> str:
        try:
            if not product_text or not product_text.strip():
                raise ValueError("Product text cannot be empty.")

            return (
                "Clean and summarize this product information for price prediction:\n\n"
                f"{product_text.strip()}"
            )

        except Exception as e:
            logger.exception("Failed to build preprocessing prompt.")
            raise RuntimeError(f"Failed to build preprocessing prompt: {e}") from e

    def preprocess_text(self, product_text: str) -> str:
        try:
            prompt = self.build_user_prompt(product_text)

            summary = self.client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            if not summary:
                raise ValueError("Ollama returned an empty product summary.")

            logger.info("Product text preprocessed successfully.")
            return summary.strip()

        except Exception as e:
            logger.exception("Failed to preprocess product text.")
            raise RuntimeError(f"Failed to preprocess product text: {e}") from e

    def preprocess_item(self, item):
        try:
            product_text = item.full or item.build_full_text()
            item.summary = self.preprocess_text(product_text)

            logger.info("Item preprocessed successfully: %s", item.title)
            return item

        except Exception as e:
            logger.exception("Failed to preprocess item: %s", getattr(item, "title", "Unknown"))
            raise RuntimeError(f"Failed to preprocess item: {e}") from e