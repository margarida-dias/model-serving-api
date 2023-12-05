"""`SERVINGAPI.logger` module."""

import logging
import logging.config


def setup_logger(level: str):
    """Define logging configuration."""
    config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(name)s %(levelname)-4s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": level,
            }
        },
        "root": {"handlers": ["console"], "level": level},
        "loggers": {
            "fastapi": {"propagate": True},
            "mlserver": {"propagate": True},
            "uvicorn": {"propagate": True},
            "uvicorn.access": {"propagate": True},
        },
    }

    logging.config.dictConfig(config)
