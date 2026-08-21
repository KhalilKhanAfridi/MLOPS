import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------- Logging setup -------------------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> int:
    """Load max_features from a YAML params file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
        logger.info("Parameters retrieved from %s | max_features=%s", params_path, max_features)
        return max_features
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


def load_data(path: str) -> pd.DataFrame:
    """Load a processed CSV file and fill missing values."""
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace=True)
        logger.info("Loaded data from %s | shape=%s", path, df.shape)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("No data found in file: %s", path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV %s: %s", path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading %s: %s", path, e)
        raise


def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Bag-of-Words (CountVectorizer) to train and test content columns."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info(
            "Bag-of-Words applied | train_shape=%s test_shape=%s max_features=%s",
            train_df.shape, test_df.shape, max_features
        )
        return train_df, test_df

    except KeyError as e:
        logger.error("Expected column missing in data (need 'content' and 'sentiment'): %s", e)
        raise
    except ValueError as e:
        logger.error("Value error during vectorization (check for empty/invalid text data): %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while applying Bag-of-Words: %s", e)
        raise


def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save the feature-engineered train and test sets to disk."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_path = os.path.join(data_path, "train_bog.csv")
        test_path = os.path.join(data_path, "test_bog.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Saved train features to %s | shape=%s", train_path, train_df.shape)
        logger.info("Saved test features to %s | shape=%s", test_path, test_df.shape)
    except PermissionError as e:
        logger.error("Permission denied while writing to %s: %s", data_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving data to %s: %s", data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise


def main() -> None:
    try:
        max_features = load_params('params.yaml')

        train_data = load_data("./data/processed/train_processed.csv")
        test_data = load_data("./data/processed/test_processed.csv")

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        data_path = os.path.join("data", "features")
        save_data(data_path, train_df, test_df)

        logger.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.critical("Feature engineering pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()