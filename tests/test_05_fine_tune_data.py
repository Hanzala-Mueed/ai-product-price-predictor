import json

from pricer.fine_tune_data import FineTuneDataBuilder


def test_load_rows_from_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        '"Title: Test Product\nCategory: Electronics",99\n',
        encoding="utf-8",
    )

    builder = FineTuneDataBuilder(csv_path, tmp_path)
    rows = builder.load_rows()

    assert len(rows) == 1
    assert rows[0]["price"] == "99"
    assert "Title: Test Product" in rows[0]["product_text"]


def test_format_example():
    builder = FineTuneDataBuilder("fake.csv", "fake_output")

    row = {
        "product_text": "Title: Test Product\nCategory: Electronics",
        "price": "99",
    }

    example = builder.format_example(row)

    assert "messages" in example
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][2]["role"] == "assistant"
    assert example["messages"][2]["content"] == "99"


def test_build_creates_jsonl_files(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        '"Title: Product One\nCategory: Electronics",99\n'
        '"Title: Product Two\nCategory: Home",49\n'
        '"Title: Product Three\nCategory: Office",25\n'
        '"Title: Product Four\nCategory: Tools",75\n'
        '"Title: Product Five\nCategory: Music",120\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "processed"

    builder = FineTuneDataBuilder(
        input_csv=csv_path,
        output_dir=output_dir,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    paths = builder.build()

    assert paths["train"].exists()
    assert paths["validation"].exists()
    assert paths["test"].exists()

    with paths["train"].open("r", encoding="utf-8") as file:
        first_line = json.loads(file.readline())

    assert "messages" in first_line