import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    """
    Handles reading and loading configuration settings from YAML resource files.
    """

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Loads and parses a YAML configuration file.

        Args:
            file_path (str): Absolute or relative filesystem path.

        Returns:
            Dict[str, Any]: Parsed configuration values.

        Raises:
            FileNotFoundError: If the config path does not exist.
            yaml.YAMLError: If parsing configuration fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f) or {}
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
