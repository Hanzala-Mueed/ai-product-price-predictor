import pytest
from pydantic import ValidationError
from pricer.items import Item, PREFIX


def test_item_creation_with_required_fields():
    item = Item(
        title="Wireless Bluetooth Headphones",
        category="Electronics",
        price=59.99,
    )

    assert item.title == "Wireless Bluetooth Headphones"
    assert item.category == "Electronics"
    assert item.price == 59.99


def test_item_strips_whitespace():
    item = Item(
        title="  Smart Watch  ",
        category="  Electronics  ",
        brand="  FitBrand  ",
    )

    assert item.title == "Smart Watch"
    assert item.category == "Electronics"
    assert item.brand == "FitBrand"


def test_item_rejects_empty_title():
    with pytest.raises(ValidationError):
        Item(title="   ", category="Electronics")


def test_item_rejects_negative_price():
    with pytest.raises(ValidationError):
        Item(title="Camera", category="Electronics", price=-10)


def test_build_full_text():
    item = Item(
        title="Gaming Mouse",
        category="Electronics",
        brand="LogiTech",
        description="A fast wireless gaming mouse.",
        features="RGB lighting, ergonomic design",
    )

    full_text = item.build_full_text()

    assert "Title: Gaming Mouse" in full_text
    assert "Category: Electronics" in full_text
    assert "Brand: LogiTech" in full_text
    assert "Description: A fast wireless gaming mouse." in full_text
    assert "Features: RGB lighting, ergonomic design" in full_text


def test_make_training_prompt_requires_price():
    item = Item(title="Laptop Stand", category="Office")

    with pytest.raises(RuntimeError):
        item.make_training_prompt()


def test_make_training_prompt():
    item = Item(
        title="USB-C Hub",
        category="Electronics",
        price=29.99,
        description="A compact USB-C hub with HDMI and USB ports.",
    )

    prompt = item.make_training_prompt()

    assert "What does this product cost" in prompt
    assert "USB-C Hub" in prompt
    assert f"{PREFIX}30.00" in prompt


def test_make_prediction_prompt():
    item = Item(
        title="Desk Lamp",
        category="Home Office",
        description="LED desk lamp with adjustable brightness.",
    )

    prompt = item.make_prediction_prompt()

    assert "Desk Lamp" in prompt
    assert prompt.endswith(PREFIX)