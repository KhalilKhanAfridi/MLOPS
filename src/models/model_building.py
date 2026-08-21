import logging
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

# ------------------------- Logging setup -------------------------
logger = logging.getLogger("model_building")
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


def load_params(params_path: str) -> dict:
    """Load model_building params from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        n_estimators = params['model_building']['n_estimators']
        learning_rate = params['model_building']['learning_rate']
        logger.info(
            "Parameters retrieved from %s | n_estimators=%s learning_rate=%s",
            params_path, n_estimators, learning_rate
        )
        return params
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
    """Load the Bag-of-Words feature CSV."""
    try:
        df = pd.read_csv(path)
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


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame into features (all but last column) and labels (last column)."""
    try:
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values
        logger.info("Split data into X=%s y=%s", X.shape, y.shape)
        return X, y
    except Exception as e:
        logger.error("Error splitting features/labels: %s", e)
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    learning_rate: float
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier."""
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return xgb_model
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model: xgb.XGBClassifier, model_path: str) -> None:
    """Save the trained model to disk as a pickle file."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", model_path)
    except PermissionError as e:
        logger.error("Permission denied while writing to %s: %s", model_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving model to %s: %s", model_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving model: %s", e)
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')

        train_data = load_data("./data/features/train_bog.csv")
        X_train, y_train = split_features_labels(train_data)

        xgb_model = train_model(
            X_train,
            y_train,
            n_estimators=params['model_building']['n_estimators'],
            learning_rate=params['model_building']['learning_rate']
        )

        save_model(xgb_model, 'model.pkl')

        logger.info("Model building pipeline completed successfully.")

    except Exception as e:
        logger.critical("Model building pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()