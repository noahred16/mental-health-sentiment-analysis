import kagglehub
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    # print(df.head())

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


def get_train_test_split(df, test_size=0.2, random_state=42):
    """Get consistent train/test split for the dataset"""

    DEBUG = True  # Comment out for full dataset usage
    if DEBUG:
        SAMPLE_FRACTION = 0.0025  # Use fraction of the data for quick testing
        print(f"\nDEBUG Sampling {SAMPLE_FRACTION*100}% of data for faster testing...")
        df = df.groupby("status", group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=42)
        )

    return train_test_split(
        df["processed_text"].values,
        df["status"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["status"].values,
    )


def describe_data(df):
    """Print basic statistics of the dataset"""
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df["status"].value_counts())

    # pie chart distribution of the status column
    # counts = df["status"].value_counts()
    # counts.plot(kind="pie", autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(df)):,})', figsize=(10, 8), textprops={'fontsize': 9}, pctdistance=0.85)
    # plt.title("Class Distribution")
    # plt.ylabel("")  # Hide the y-label
    # # save to metrics
    # plt.savefig("metrics/class_distribution.png", bbox_inches='tight')
    # plt.close()

    # get total number of unique words in the dataset
    # unique_words = set()
    # for text in df["processed_text"]:
    #     unique_words.update(text.split())
    # print("Total unique words in the dataset:", len(unique_words))
