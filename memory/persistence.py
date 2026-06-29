import json
import os
import logging
from memory.models import SessionState

logger = logging.getLogger(__name__)

class MemoryPersistence:
    """
    Handles saving and loading the memory SessionState to and from JSON files.
    """
    @staticmethod
    def save_session(state: SessionState, file_path: str) -> bool:
        """
        Saves the SessionState to the specified file path in JSON format.
        """
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            
            data = state.to_dict()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved session state to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session state to {file_path}: {e}")
            return False

    @staticmethod
    def load_session(file_path: str) -> SessionState:
        """
        Loads the SessionState from the specified file path.
        Returns a clean SessionState if file not found or load fails.
        """
        if not os.path.exists(file_path):
            logger.info(f"Session file {file_path} not found. Returning clean SessionState.")
            return SessionState()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Successfully loaded session state from {file_path}")
            return SessionState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session state from {file_path}: {e}. Returning clean SessionState.")
            return SessionState()
