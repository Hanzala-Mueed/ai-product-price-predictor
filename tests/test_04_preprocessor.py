import pytest
from pricer.items import Item
from pricer.preprocessor import ProductPreprocessor


class FakeOllamaClient:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "Title: Wireless Bluetooth Headphones\n"
            "Category: Electronics\n"
            "Brand: SoundPro\n"
            "Description: Wireless headphones designed for everyday listening.\n"
            "Details: Includes Bluetooth connectivity, soft ear cushions, and long battery life."
        )


class EmptyFakeOllamaClient:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return ""


def test_build_user_prompt():
    preprocessor = ProductPreprocessor(client=FakeOllamaClient())

    prompt = preprocessor.build_user_prompt("Wireless headphones with Bluetooth.")

    assert "Clean and summarize" in prompt
    assert "Wireless headphones with Bluetooth." in prompt


def test_build_user_prompt_rejects_empty_text():
    preprocessor = ProductPreprocessor(client=FakeOllamaClient())

    with pytest.raises(RuntimeError):
        preprocessor.build_user_prompt("   ")


def test_preprocess_text_returns_summary():
    preprocessor = ProductPreprocessor(client=FakeOllamaClient())

    summary = preprocessor.preprocess_text("Wireless headphones with Bluetooth.")

    assert "Title:" in summary
    assert "Category:" in summary
    assert "Description:" in summary


def test_preprocess_text_rejects_empty_ollama_response():
    preprocessor = ProductPreprocessor(client=EmptyFakeOllamaClient())

    with pytest.raises(RuntimeError):
        preprocessor.preprocess_text("Wireless headphones with Bluetooth.")


def test_preprocess_item_updates_summary():
    item = Item(
        title="Wireless Bluetooth Headphones",
        category="Electronics",
        brand="SoundPro",
        description="Wireless headphones with Bluetooth and long battery life.",
    )

    preprocessor = ProductPreprocessor(client=FakeOllamaClient())
    processed_item = preprocessor.preprocess_item(item)

    assert processed_item.summary is not None
    assert "Wireless Bluetooth Headphones" in processed_item.summary