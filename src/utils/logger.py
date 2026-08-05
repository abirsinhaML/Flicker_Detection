"""Application logging configuration."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def configure_logging(config_path: str | Path) -> None:
    """Configure standard-library logging from a YAML dictionary config."""
    config_file = Path(config_path)
    with config_file.open(encoding="utf-8") as config_stream:
        config: dict[str, Any] = yaml.safe_load(config_stream)
    logging.config.dictConfig(config)
