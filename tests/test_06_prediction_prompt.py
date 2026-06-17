import pytest

from pricer.items import Item
from pricer.predictor import PricePredictor


class FakeOllamaClient:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "59"


class BadOllamaClient:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "I do not know."


def test_build_prediction_prompt():
    item = Item(
        title="Wireless Mouse",
        category="Electronics",
        brand="LogiTech",
        description="A wireless mouse with ergonomic design.",
    )

    predictor = PricePredictor(client=FakeOllamaClient())
    prompt = predictor.build_prompt(item)

    assert "Wireless Mouse" in prompt
    assert "Price is $" in prompt


def test_extract_price_from_clean_number():
    predictor = PricePredictor(client=FakeOllamaClient())

    price = predictor.extract_price("59")

    assert price == 59


def test_extract_price_from_dollar_response():
    predictor = PricePredictor(client=FakeOllamaClient())

    price = predictor.extract_price("$1,299.99")

    assert price == 1299.99


def test_extract_price_rejects_invalid_response():
    predictor = PricePredictor(client=FakeOllamaClient())

    with pytest.raises(RuntimeError):
        predictor.extract_price("No price available")


def test_predict_returns_price():
    item = Item(
        title="Bluetooth Speaker",
        category="Electronics",
        summary=(
            "Title: Bluetooth Speaker\n"
            "Category: Electronics\n"
            "Brand: SoundPro\n"
            "Description: Portable speaker for music playback.\n"
            "Details: Bluetooth connectivity and rechargeable battery."
        ),
    )

    predictor = PricePredictor(client=FakeOllamaClient())
    price = predictor.predict(item)

    assert price == 59


def test_predict_fails_for_bad_response():
    item = Item(
        title="Bluetooth Speaker",
        category="Electronics",
        summary="Portable Bluetooth speaker with rechargeable battery.",
    )

    predictor = PricePredictor(client=BadOllamaClient())

    with pytest.raises(RuntimeError):
        predictor.predict(item)