from training.import_adapter_to_ollama import (
    ADAPTER_DIR,
    MODELFILE_PATH,
    FINETUNED_MODEL_NAME,
)


def test_adapter_directory_path():
    assert "models" in str(ADAPTER_DIR)
    assert "adapters" in str(ADAPTER_DIR)
    assert "product_price_lora" in str(ADAPTER_DIR)


def test_modelfile_path():
    assert MODELFILE_PATH.name == "Modelfile"


def test_finetuned_model_name():
    assert FINETUNED_MODEL_NAME == "product-price-predictor-finetuned"