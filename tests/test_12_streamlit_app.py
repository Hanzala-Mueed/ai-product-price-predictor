from pathlib import Path

import app


def test_app_file_exists():
    assert Path("app.py").exists()


def test_app_has_main_function():
    assert hasattr(app, "main")


def test_build_item_from_form_data():
    item = app.build_item(
        title="Wireless Mouse",
        category="Electronics",
        brand="LogiTech",
        description="A wireless mouse for everyday use.",
        features="Ergonomic design and USB receiver.",
    )

    assert item.title == "Wireless Mouse"
    assert item.category == "Electronics"
    assert item.brand == "LogiTech"