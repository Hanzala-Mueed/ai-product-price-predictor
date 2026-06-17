from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

APP_NAME = "AI Product Price Predictor"
APP_VERSION = "1.0.0"