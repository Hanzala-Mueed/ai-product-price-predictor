import logging
from logging.handlers import RotatingFileHandler
from config.settings import LOG_DIR, LOG_FILE


def setup_logging() -> logging.Logger:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("price_predictor")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Logging system initialized successfully.")
        return logger

    except Exception as e:
        raise RuntimeError(f"Failed to configure logging: {e}") from e