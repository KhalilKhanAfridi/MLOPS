import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split
import yaml

# ------------------------- Logging setup -------------------------
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)  # logger itself accepts everything DEBUG and above

# Console handler -> shows everything (DEBUG and above) while script runs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler -> only saves ERROR and above to a persistent log file
file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

# Formatter -> same message format for both handlers
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    """Load test_size from a YAML params file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.info("Parameters retrieved from %s", params_path)
        return test_size
    except FileNotFoundError:
        logger.error("Params file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parsing error in %s: %s", params_path, e)
        raise
    except KeyError as e:
        logger.error("Missing key %s in params file %s", e, params_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params: %s", e)
        raise


def read_data(url: str) -> pd.DataFrame:
    """Read CSV data from a URL or local path."""
    try:
        df = pd.read_csv(url)
        logger.info("Data read successfully from %s | shape=%s", url, df.shape)
        return df
    except pd.errors.EmptyDataError:
        logger.error("No data found at source: %s", url)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV from %s: %s", url, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while reading data from %s: %s", url, e)
        raise


def processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and encode sentiment data."""
    try:
        df = df.drop(columns=['tweet_id'])
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

        if final_df.empty:
            raise ValueError("No rows left after filtering for 'happiness'/'sadness' sentiments.")

        logger.info("Data processed successfully | shape=%s", final_df.shape)
        return final_df
    except KeyError as e:
        logger.error("Expected column missing during processing: %s", e)
        raise
    except ValueError as e:
        logger.error("Data validation failed: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during data processing: %s", e)
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test datasets to disk."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info("Train data saved to %s | shape=%s", train_path, train_data.shape)
        logger.info("Test data saved to %s | shape=%s", test_path, test_data.shape)
    except PermissionError as e:
        logger.error("Permission denied while writing to %s: %s", data_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving data to %s: %s", data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise


def main():
    try:
        test_size = load_params('params.yaml')

        df = read_data(
            'https://raw.githubusercontent.com/campusx-official/'
            'jupyter-masterclass/main/tweet_emotions.csv'
        )

        final_df = processed_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )

        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.critical("Data ingestion pipeline failed: %s", e)
        sys.exit(1)  # non-zero exit so DVC/CI knows this stage failed


if __name__ == "__main__":
    main()