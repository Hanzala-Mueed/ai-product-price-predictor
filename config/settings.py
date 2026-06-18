from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

OLLAMA_BASE_URL = "http://localhost:11434"

BASE_OLLAMA_MODEL = "llama3.2:latest"
CUSTOM_OLLAMA_MODEL = "product-price-predictor"

OLLAMA_MODEL = CUSTOM_OLLAMA_MODEL

APP_NAME = "AI Product Price Predictor"
APP_VERSION = "1.0.0"