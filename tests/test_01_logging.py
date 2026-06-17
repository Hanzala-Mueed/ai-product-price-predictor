from config.logging_config import setup_logging
from config.settings import LOG_FILE


def test_logging_setup_creates_logger():
    logger = setup_logging()

    assert logger is not None
    assert logger.name == "price_predictor"


def test_logging_writes_to_file():
    logger = setup_logging()
    logger.info("Test logging message")

    assert LOG_FILE.exists()

    content = LOG_FILE.read_text(encoding="utf-8")
    assert "Test logging message" in content