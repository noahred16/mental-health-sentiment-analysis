import kagglehub
import pandas as pd
import re
import os

FILE_PATH = "data/mental_health_sentiment_analysis.csv"


# download the dataset from source anb copy to data directory
# https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data
def setup():
    dataset_path = kagglehub.dataset_download(
        "suchintikasarkar/sentiment-analysis-for-mental-health"
    )

    # Access your file
    file_path = os.path.join(dataset_path, "Combined Data.csv")
    df = pd.read_csv(file_path)

    # Print first 5 rows
    print(df.head())

    if os.path.exists(FILE_PATH):
        return

    # Copy dataset to the data directory
    os.makedirs("data", exist_ok=True)
    df.to_csv(FILE_PATH, index=False)

    print("Dataset saved to:", FILE_PATH)


def preprocess_text(text):
    """Basic text preprocessing"""
    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to string in case of other data types
    text = str(text)

    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def load_data():
    setup()
    """Load and preprocess the dataset"""
    df = pd.read_csv(FILE_PATH)

    # Preprocess the text column
    df["processed_text"] = df["statement"].apply(preprocess_text)

    # Drop rows with empty text
    df = df[df["processed_text"].str.strip() != ""]

    return df
