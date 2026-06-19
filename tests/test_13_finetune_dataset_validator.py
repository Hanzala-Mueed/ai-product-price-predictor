import json

import pytest

from training.validate_finetune_dataset import FineTuneDatasetValidator


def make_valid_example():
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI product pricing assistant.",
            },
            {
                "role": "user",
                "content": "What does this product cost?\n\nTitle: Test Product\n\nPrice is $",
            },
            {
                "role": "assistant",
                "content": "99",
            },
        ]
    }


def test_validate_message_accepts_valid_message(tmp_path):
    validator = FineTuneDatasetValidator(tmp_path / "fake.jsonl")

    message = {
        "role": "user",
        "content": "Test content",
    }

    assert validator.validate_message(message) is True


def test_validate_message_rejects_invalid_role(tmp_path):
    validator = FineTuneDatasetValidator(tmp_path / "fake.jsonl")

    message = {
        "role": "bad_role",
        "content": "Test content",
    }

    with pytest.raises(RuntimeError):
        validator.validate_message(message)


def test_validate_example_accepts_valid_example(tmp_path):
    validator = FineTuneDatasetValidator(tmp_path / "fake.jsonl")

    assert validator.validate_example(make_valid_example()) is True


def test_validate_example_rejects_non_numeric_price(tmp_path):
    validator = FineTuneDatasetValidator(tmp_path / "fake.jsonl")

    example = make_valid_example()
    example["messages"][2]["content"] = "ninety nine"

    with pytest.raises(RuntimeError):
        validator.validate_example(example)


def test_validate_file_accepts_valid_jsonl(tmp_path):
    jsonl_path = tmp_path / "valid.jsonl"

    jsonl_path.write_text(
        json.dumps(make_valid_example()) + "\n",
        encoding="utf-8",
    )

    validator = FineTuneDatasetValidator(jsonl_path)
    report = validator.validate_file()

    assert report["total"] == 1
    assert report["valid"] == 1
    assert report["invalid"] == 0
    assert report["is_valid"] is True


def test_validate_file_reports_invalid_jsonl(tmp_path):
    jsonl_path = tmp_path / "invalid.jsonl"

    jsonl_path.write_text(
        "{bad json}\n",
        encoding="utf-8",
    )

    validator = FineTuneDatasetValidator(jsonl_path)
    report = validator.validate_file()

    assert report["total"] == 1
    assert report["valid"] == 0
    assert report["invalid"] == 1
    assert report["is_valid"] is False