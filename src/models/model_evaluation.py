import json
import logging
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# ------------------------- Logging setup -------------------------
logger = logging.getLogger("model_evaluation")
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


def load_model(model_path: str) -> Any:
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.info("Model loaded from %s", model_path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        raise
    except pickle.UnpicklingError as e:
        logger.error("Failed to unpickle model from %s: %s", model_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model: %s", e)
        raise


def load_data(path: str) -> pd.DataFrame:
    """Load the Bag-of-Words test feature CSV."""
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


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Run predictions and compute evaluation metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

        logger.info("Evaluation metrics computed: %s", metrics)
        return metrics
    except ValueError as e:
        logger.error("Value error during evaluation (check label format/shape): %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during model evaluation: %s", e)
        raise


def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """Save metrics dictionary to a JSON file."""
    try:
        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.info("Metrics saved to %s", metrics_path)
    except PermissionError as e:
        logger.error("Permission denied while writing to %s: %s", metrics_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving metrics to %s: %s", metrics_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving metrics: %s", e)
        raise


def main() -> None:
    try:
        xgb_model = load_model("model.pkl")

        test_data = load_data("./data/features/test_tfidf.csv")
        X_test, y_test = split_features_labels(test_data)

        metrics = evaluate_model(xgb_model, X_test, y_test)

        for name, value in metrics.items():
            logger.info("%s: %s", name.capitalize(), value)

        save_metrics(metrics, "metrics.json")

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.critical("Model evaluation pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()