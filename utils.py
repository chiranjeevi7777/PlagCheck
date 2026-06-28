import logging
import sys
import nltk
from pathlib import Path

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PlagCheck")

def init_nltk():
    """Download required NLTK resources if not already present."""
    try:
        # Check if tokenizer is already downloaded
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' package is already present.")
    except LookupError:
        logger.info("NLTK 'punkt' package not found. Downloading...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK 'punkt' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK 'punkt': {e}")
            
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK 'punkt_tab' downloaded successfully.")
        except Exception:
            pass

# Initialize on import
init_nltk()
