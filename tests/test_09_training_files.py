from pathlib import Path

from training.train_lora_unsloth import (
    BASE_DIR,
    TRAIN_FILE,
    VALIDATION_FILE,
    OUTPUT_DIR,
    BASE_MODEL_NAME,
    validate_training_files,
)


def test_training_paths_are_inside_project():
    assert BASE_DIR.exists()
    assert "data" in str(TRAIN_FILE)
    assert "data" in str(VALIDATION_FILE)
    assert "models" in str(OUTPUT_DIR)


def test_base_model_is_llama_3_2():
    assert "Llama-3.2" in BASE_MODEL_NAME


def test_validate_training_files():
    assert validate_training_files() is True