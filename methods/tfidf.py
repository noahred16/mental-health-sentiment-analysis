import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# load preprocessed data
df = utils.load_data()

# encode labels as integers
df["label"] = df["status"].astype("category").cat.codes
label_mapping = dict(enumerate(df["status"].astype("category").cat.categories))
print("Label mapping:", label_mapping)

# 80% train, 10% val, 10% test split
X_train, X_val, X_test, y_train, y_val, y_test = utils.get_train_val_test_split(
    df, val_size=0.1, test_size=0.1, random_state=42
)


def train_tfidf_model(X_train, y_train):
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
    )

    # Hyperparameter Grid
    param_grid = {
        "tfidf__max_features": [3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 10],
        "clf__solver": ["lbfgs"],
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1
    )
    grid.fit(X_train, y_train)
