import re
from typing import Optional

from pricer.items import Item
from pricer.ollama_client import OllamaClient
from config.logging_config import setup_logging

from config.settings import PREDICTOR_OLLAMA_MODEL, PREPROCESSOR_OLLAMA_MODEL

logger = setup_logging()


PRICE_SYSTEM_PROMPT = """You are an AI product price prediction assistant.

Your task:
Estimate the one realistic marketplace price of the product.

Rules:
- Respond only with one price number.
- Do not include a dollar sign.
- Do not include explanation.
- Round to the nearest dollar.
- If unsure, make the best realistic estimate.
- No ranges, only one price.

Example:
47
"""

class PricePredictor:
    def __init__(self, client: Optional[OllamaClient] = None, fallback_client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient(model=PREDICTOR_OLLAMA_MODEL)
        self.fallback_client = fallback_client or OllamaClient(model=PREPROCESSOR_OLLAMA_MODEL)
# class PricePredictor:
#     def __init__(self, client: Optional[OllamaClient] = None):
#         self.client = client or OllamaClient(model=PREDICTOR_OLLAMA_MODEL)

    def build_prompt(self, item: Item) -> str:
        try:
            product_text = item.summary or item.full or item.build_full_text()

            prompt = (
                "What does this product cost to the nearest dollar?\n\n"
                f"{product_text}\n\n"
                "Price is $"
            )

            logger.info("Prediction prompt created for item: %s", item.title)
            return prompt

        except Exception as e:
            logger.exception("Failed to build prediction prompt.")
            raise RuntimeError(f"Failed to build prediction prompt: {e}") from e

    def extract_price(self, response: str) -> float:
        try:
            if not response or not response.strip():
                raise ValueError("Empty response received from model.")

            cleaned = response.replace(",", "").replace("$", "").strip()
            match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)

            if not match:
                raise ValueError(f"No price found in model response: {response}")

            price = float(match.group())

            if price < 0:
                raise ValueError("Predicted price cannot be negative.")

            logger.info("Extracted predicted price: %s", price)
            return price

        except Exception as e:
            logger.exception("Failed to extract price from model response.")
            raise RuntimeError(f"Failed to extract price from model response: {e}") from e

    # def predict(self, item: Item) -> float:
    #     try:
    #         prompt = self.build_prompt(item)

    #         response = self.client.generate(
    #             prompt=prompt,
    #             system_prompt=PRICE_SYSTEM_PROMPT,
    #         )

    #         price = self.extract_price(response)

    #         logger.info("Price prediction completed for item: %s", item.title)
    #         return price

    #     except Exception as e:
    #         logger.exception("Failed to predict product price.")
    #         raise RuntimeError(f"Failed to predict product price: {e}") from e

    def predict(self, item: Item) -> float:
        prompt = self.build_prompt(item)

        try:
            response = self.client.generate(
                prompt=prompt,
                system_prompt=PRICE_SYSTEM_PROMPT,
            )
            return self.extract_price(response)

        except Exception as e:
            logger.warning(
                "Primary prediction failed using fine-tuned model. Falling back to base model. Error: %s",
                e,
            )

            try:
                response = self.fallback_client.generate(
                    prompt=prompt,
                    system_prompt=PRICE_SYSTEM_PROMPT,
                )
                price = self.extract_price(response)

                logger.info("Fallback price prediction completed for item: %s", item.title)
                return price

            except Exception as fallback_error:
                logger.exception("Both primary and fallback price prediction failed.")
                raise RuntimeError(
                    f"Failed to predict product price with both models: {fallback_error}"
                ) from fallback_error