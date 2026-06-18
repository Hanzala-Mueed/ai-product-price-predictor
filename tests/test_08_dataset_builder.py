import pytest

from pricer.dataset_builder import DatasetBuilder
from pricer.items import Item


def test_extract_title():
    builder = DatasetBuilder("fake.csv")

    title = builder.extract_title(
        "Title: Wireless Mouse\nCategory: Electronics\nDescription: Test"
    )

    assert title == "Wireless Mouse"


def test_extract_category():
    builder = DatasetBuilder("fake.csv")

    category = builder.extract_category(
        "Title: Wireless Mouse\nCategory: Electronics\nDescription: Test"
    )

    assert category == "Electronics"


def test_load_items_from_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        '"Title: Product One\nCategory: Electronics\nDescription: Test",99\n'
        '"Title: Product Two\nCategory: Home\nDescription: Test",49\n',
        encoding="utf-8",
    )

    builder = DatasetBuilder(csv_path)
    items = builder.load_items()

    assert len(items) == 2
    assert isinstance(items[0], Item)
    assert items[0].title == "Product One"
    assert items[0].category == "Electronics"
    assert items[0].price == 99


def test_load_items_with_limit(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        '"Title: Product One\nCategory: Electronics",99\n'
        '"Title: Product Two\nCategory: Home",49\n',
        encoding="utf-8",
    )

    builder = DatasetBuilder(csv_path)
    items = builder.load_items(limit=1)

    assert len(items) == 1


def test_load_items_rejects_missing_file():
    builder = DatasetBuilder("missing.csv")

    with pytest.raises(RuntimeError):
        builder.load_items()