# src/utils/helpers.py
import logging
import json
import yaml

def setup_logging(log_level=logging.INFO):
    """Sets up basic logging."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_jsonl(file_path):
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {file_path}: {e}")
        return [] # Or raise error

def save_jsonl(data, file_path):
    """Saves data to a JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logging.info(f"Saved {len(data)} records to {file_path}")
    except IOError as e:
        logging.error(f"Error writing to {file_path}: {e}")

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        return {}

# Initialize logger when module is imported
logger = setup_logging()

# --- Add other common helper functions as needed ---