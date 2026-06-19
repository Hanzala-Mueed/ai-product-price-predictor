

from pathlib import Path

from training.colab_train_lora_unsloth import (
    BASE_MODEL_NAME,
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    install_dependencies,
    training_code,
    zip_adapter_code,
)


def test_colab_base_model_is_llama_3_2():
    assert "Llama-3.2" in BASE_MODEL_NAME


def test_colab_output_dir_name():
    assert OUTPUT_DIR == "product_price_lora"


def test_max_sequence_length():
    assert MAX_SEQ_LENGTH == 2048


def test_install_dependencies_contains_unsloth():
    content = install_dependencies()

    assert "unsloth" in content


def test_training_code_contains_dataset_files():
    content = training_code()

    assert "train.jsonl" in content
    assert "validation.jsonl" in content
    assert "SFTTrainer" in content


def test_zip_adapter_code():
    content = zip_adapter_code()

    assert "product_price_lora.zip" in content