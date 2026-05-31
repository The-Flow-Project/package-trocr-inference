"""Configure and expose the shared inference logger.

This module loads the YAML logging configuration bundled with the package,
ensures that the local log output directory exists, and exposes the central
``inference_logger`` instance used throughout the inference workflow.
"""
import logging.config
import yaml
import os

# Ensure log directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Load logging configuration
current_dir = os.path.dirname(os.path.realpath(__file__))
logging_config = os.path.join(current_dir, "logging_config.yaml")
with open(logging_config, "r") as file:
    config = yaml.safe_load(file)
    logging.config.dictConfig(config)

# Central logger accessible throughout the application
logger = logging.getLogger("inference_logger")
