import yaml
import logging
import os
import pickle
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="LongEval Scientific Retrieval Pipeline")
    parser.add_argument("--config", type=str, default="abstract_config.yaml", help="Path to the configuration file.")
    return parser.parse_args()

def load_config(config_path="abstract_config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")

        BASE_DIR = os.path.dirname(os.path.abspath(config_path))

        # Normalize key paths relative to the config file
        if "data_dir" in config:
            config["data_dir"] = os.path.normpath(os.path.join(BASE_DIR, config["data_dir"]))
        if "output_dir" in config:
            config["output_dir"] = os.path.normpath(os.path.join(BASE_DIR, config["output_dir"]))
        if config.get("do_finetune", False) and "st_model_output_path" in config:
            config["st_model_output_path"] = os.path.normpath(os.path.join(BASE_DIR, config["st_model_output_path"]))

        # Create output directories
        os.makedirs(config["output_dir"], exist_ok=True)
        if config.get("do_finetune", False):
            os.makedirs(config["st_model_output_path"], exist_ok=True)

        return config

    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def save_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving pickle to {file_path}: {e}")
        raise

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Data loaded from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Pickle file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading pickle from {file_path}: {e}")
        raise