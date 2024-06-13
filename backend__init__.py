# backend/__init__.py

import logging
import os

# Configure logging for the backend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the data directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

logger.info("Backend initialization complete.")
