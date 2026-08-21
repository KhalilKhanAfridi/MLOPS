import os
import re
import string
import logging
from typing import List

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------- Logging setup -------------------------
logger = logging.getLogger("data_preprocessing")
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


# ------------------------- NLTK setup -------------------------
def download_nltk_resources() -> None:
    """Download required NLTK corpora."""
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logger.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logger.error("Failed to download NLTK resources: %s", e)
        raise


# ------------------------- Text cleaning functions -------------------------
def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)
    except Exception as e:
        logger.error("Error during lemmatization for text=%r: %s", text, e)
        raise


def remove_stop_words(text: str) -> str:
    """Remove English stopwords from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        words = [word for word in str(text).split() if word not in stop_words]
        return " ".join(words)
    except Exception as e:
        logger.error("Error removing stop words for text=%r: %s", text, e)
        raise


def removing_numbers(text: str) -> str:
    """Remove numeric tokens from the text."""
    try:
        return " ".join([word for word in text.split() if not word.isdigit()])
    except Exception as e:
        logger.error("Error removing numbers for text=%r: %s", text, e)
        raise


def lower_case(text: str) -> str:
    """Lowercase every word in the text."""
    try:
        words = text.split()
        words = [word.lower() for word in words]
        return " ".join(words)
    except Exception as e:
        logger.error("Error lowercasing text=%r: %s", text, e)
        raise


def removing_punctuations(text: str) -> str:
    """Remove punctuation and collapse extra whitespace."""
    try:
        text = re.sub(
            '[%s]' % re.escape(string.punctuation), ' ', text
        )
        text = text.replace('`', "")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error("Error removing punctuations for text=%r: %s", text, e)
        raise


def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    except Exception as e:
        logger.error("Error removing URLs for text=%r: %s", text, e)
        raise


def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Replace texts with fewer than 3 words with NaN."""
    try:
        df = df.copy()
        mask = df['content'].apply(lambda x: len(str(x).split()) < 3)
        df.loc[mask, 'content'] = np.nan
        logger.info("Removed %d small sentences.", mask.sum())
        return df
    except KeyError as e:
        logger.error("Expected 'content' column missing: %s", e)
        raise
    except Exception as e:
        logger.error("Error removing small sentences: %s", e)
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text-normalization pipeline to the 'content' column."""
    try:
        df = df.copy()
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.info("Text normalization completed | shape=%s", df.shape)
        return df
    except KeyError as e:
        logger.error("Expected 'content' column missing during normalization: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during text normalization: %s", e)
        raise


# ------------------------- I/O functions -------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
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


def save_data(df: pd.DataFrame, data_path: str, filename: str) -> None:
    """Save a DataFrame to a CSV file inside data_path."""
    try:
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, filename)
        df.to_csv(file_path, index=False)
        logger.info("Saved processed data to %s | shape=%s", file_path, df.shape)
    except PermissionError as e:
        logger.error("Permission denied while writing to %s: %s", data_path, e)
        raise
    except OSError as e:
        logger.error("OS error while saving data to %s: %s", data_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise


# ------------------------- Main pipeline -------------------------
def main() -> None:
    try:
        download_nltk_resources()

        train_data = load_data("./data/raw/train.csv")
        test_data = load_data("./data/raw/test.csv")

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        data_path = os.path.join("data", "processed")
        save_data(train_processed_data, data_path, "train_processed.csv")
        save_data(test_processed_data, data_path, "test_processed.csv")

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.critical("Data preprocessing pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()