import logging
import os
from typing import Dict, Any

def setup_logger(config: Dict[str, Any]) -> None:
    """
    Configures the project-wide logging format, level, and outputs.
    Ensures console handler is set to WARNING to maintain a clean terminal
    while the file handler receives the full INFO level diagnostics.
    """
    log_config = config.get("logging", {})
    level_name = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers = []

    # Console output handler (Strictly WARNING level or higher)
    if log_config.get("to_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        handlers.append(console_handler)

    # File output handler (Full diagnostics level)
    if log_config.get("to_file", True):
        file_path = log_config.get("file_path", "logs/pipeline.log")
        # Ensure log folder exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Apply configuration to root logger with full logging level
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )

    logging.getLogger(__name__).info("System logging initialized successfully.")
