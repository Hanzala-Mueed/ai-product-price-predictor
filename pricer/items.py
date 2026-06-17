from typing import Optional
from pydantic import BaseModel, Field, field_validator
from config.logging_config import setup_logging

logger = setup_logging()

PREFIX = "Price is $"
QUESTION = "What does this product cost to the nearest dollar?"


class Item(BaseModel):
    title: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    price: Optional[float] = Field(default=None, ge=0)
    brand: Optional[str] = None
    description: Optional[str] = None
    features: Optional[str] = None
    full: Optional[str] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    @field_validator("title", "category")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        try:
            value = value.strip()
            if not value:
                raise ValueError("Required text field cannot be empty.")
            return value
        except Exception as e:
            logger.exception("Failed to validate required text field.")
            raise e

    @field_validator("brand", "description", "features", "full", "summary", "prompt")
    @classmethod
    def clean_optional_text(cls, value: Optional[str]) -> Optional[str]:
        try:
            if value is None:
                return None
            cleaned = value.strip()
            return cleaned if cleaned else None
        except Exception as e:
            logger.exception("Failed to validate optional text field.")
            raise e

    def build_full_text(self) -> str:
        try:
            parts = [
                f"Title: {self.title}",
                f"Category: {self.category}",
            ]

            if self.brand:
                parts.append(f"Brand: {self.brand}")

            if self.description:
                parts.append(f"Description: {self.description}")

            if self.features:
                parts.append(f"Features: {self.features}")

            self.full = "\n".join(parts)
            logger.info("Built full product text for item: %s", self.title)
            return self.full

        except Exception as e:
            logger.exception("Failed to build full product text.")
            raise RuntimeError(f"Failed to build full product text: {e}") from e

    def make_training_prompt(self, text: Optional[str] = None) -> str:
        try:
            if self.price is None:
                raise ValueError("Price is required to create a training prompt.")

            product_text = text or self.summary or self.full or self.build_full_text()

            self.prompt = (
                f"{QUESTION}\n\n"
                f"{product_text}\n\n"
                f"{PREFIX}{round(self.price)}.00"
            )

            logger.info("Created training prompt for item: %s", self.title)
            return self.prompt

        except Exception as e:
            logger.exception("Failed to create training prompt.")
            raise RuntimeError(f"Failed to create training prompt: {e}") from e

    def make_prediction_prompt(self, text: Optional[str] = None) -> str:
        try:
            product_text = text or self.summary or self.full or self.build_full_text()

            prompt = (
                f"{QUESTION}\n\n"
                f"{product_text}\n\n"
                f"{PREFIX}"
            )

            logger.info("Created prediction prompt for item: %s", self.title)
            return prompt

        except Exception as e:
            logger.exception("Failed to create prediction prompt.")
            raise RuntimeError(f"Failed to create prediction prompt: {e}") from e

    def __repr__(self) -> str:
        price = f"${self.price}" if self.price is not None else "No price"
        return f"<{self.title} = {price}>"