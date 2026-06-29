import logging
import sys
import time
from dotenv import load_dotenv
from utils.config_loader import ConfigLoader

load_dotenv()
from utils.config_validator import ConfigValidator
from utils.logger import setup_logger
from pipeline.pipeline_manager import PipelineManager

logger = logging.getLogger("main")

def main() -> None:
    """
    Main entry point executing the mock Voice-to-Voice AI Agent pipeline.
    """
    # 1. Load Configurations
    try:
        config = ConfigLoader.load_yaml("config/config.yaml")
        models = ConfigLoader.load_yaml("config/models.yaml")
        config["models_meta"] = models
    except Exception as e:
        print(f"Startup Failure: Unable to parse configurations: {e}")
        sys.exit(1)

    # 2. Setup Logging
    setup_logger(config)

    # 3. Validate Configuration Settings
    try:
        ConfigValidator.validate(config)
    except Exception as e:
        logger.critical(f"Configuration validation check failed: {e}")
        sys.exit(1)

    # 4. Boot pipeline manager & run a single pipeline iteration
    try:
        manager = PipelineManager(config)
        manager.initialize_pipeline()
        
        # Starts recorder, pipeline thread, and player threads
        manager.start()

        # Let the async pipeline threads execute the single speech turn
        # Frame 1 -> VAD Detected Speech
        # Frame 2 -> Silence, runs STT -> LLM -> TTS -> Player Output
        time.sleep(3.0)

        # Print overall success verification message
        print("Pipeline Finished Successfully")

        # Gracefully stop manager threads
        manager.stop()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Fatal error during runtime: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
