import kagglehub
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import nltk

# Only download if not already present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

FILE_PATH = "data/mental_health_sentiment_analysis.csv"


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Handle missing values
        if pd.isna(text):
            return ""

        # Convert to string in case of other data types
        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs/links and email addresses
        text = re.sub(r"http\S+|www\S+|mailto:\S+", "", text)

        # Removes punctuation and special characters.
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Remove stop words and lemmatize
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)


# Global preprocessor instance
preprocessor = TextPreprocessor()


# download the dataset from source anb copy to data directory
# https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data
def setup():
    # Check if file already exists first
    if os.path.exists(FILE_PATH):
        return

    dataset_path = kagglehub.dataset_download(
        "suchintikasarkar/sentiment-analysis-for-mental-health"
    )

    # Access your file
    file_path = os.path.join(dataset_path, "Combined Data.csv")
    df = pd.read_csv(file_path)

    # Copy dataset to the data directory
    os.makedirs("data", exist_ok=True)
    df.to_csv(FILE_PATH, index=False)

    print("Dataset saved to:", FILE_PATH)


def load_data(preprocess=True):
    setup()
    """Load and optionally preprocess the dataset"""
    df = pd.read_csv(FILE_PATH)

    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0"])

    if preprocess:
        # Preprocess the text column
        df["processed_text"] = df["statement"].apply(preprocessor.preprocess_text)

        # Drop rows with empty text
        df = df[df["processed_text"].str.strip() != ""]

    return df


def get_train_test_split(df, test_size=0.2, random_state=42):
    """Get consistent train/test split for the dataset"""

    DEBUG = False
    # DEBUG = True  # Comment out for full dataset usage
    if DEBUG:
        SAMPLE_FRACTION = 0.0025  # Use fraction of the data for quick testing
        print(f"\nDEBUG Sampling {SAMPLE_FRACTION*100}% of data for faster testing...")
        df = df.groupby("status", group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=42),
            include_groups=False,
        )

    return train_test_split(
        df["processed_text"].values,
        df["status"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["status"].values,
    )


def get_train_val_test_split(df, val_size=0.1, test_size=0.1, random_state=42):
    """Get consistent train/validation/test split for the dataset"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["processed_text"].values,
        df["status"].values,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=df["status"].values,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_standard_split(df, random_state=42):
    """
    Standardized 70/20/10 train/validation/test split for all models

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        All using preprocessed_text for consistency across models
    """
    return get_train_val_test_split(
        df, val_size=0.2, test_size=0.1, random_state=random_state
    )


def describe_data(df):
    """Print basic statistics of the dataset"""
    print("Dataset shape:", df.shape)
    print("First 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    # print(df["status"].value_counts())

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

    # what are the top 10 most common words for each category
    # from collections import Counter
    # stop_words = {'i', 'and', 'to', 'the', 'a', 'my', 'of', 'it', 'that', 'im', 'is', 'in', 'me', 'have', 'for', 'with', 'on', 'am', 'so', 'but', 'not', 'be', 'can', 'like', 'feel', 'just', 'dont', 'get', 'was', 'are'}
    # print("\nTop 10 most common words for each category (excluding stop words):")
    # for category in df["status"].unique():
    #     category_texts = df[df["status"] == category]["processed_text"]
    #     all_words = []
    #     for text in category_texts:
    #         all_words.extend([word for word in text.split() if word not in stop_words])
    #     word_counts = Counter(all_words)
    #     print(f"\n{category}:")
    #     for word, count in word_counts.most_common(10):
    #         print(f"  {word}: {count}")
