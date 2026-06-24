from pricer.preprocessor import ProductPreprocessor
from pricer.predictor import PricePredictor
from config.settings import PREPROCESSOR_OLLAMA_MODEL, PREDICTOR_OLLAMA_MODEL


def test_preprocessor_uses_preprocessor_model():
    preprocessor = ProductPreprocessor()

    assert preprocessor.client.model == PREPROCESSOR_OLLAMA_MODEL


def test_predictor_uses_finetuned_model():
    predictor = PricePredictor()

    assert predictor.client.model == PREDICTOR_OLLAMA_MODEL