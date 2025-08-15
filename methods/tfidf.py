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

    return grid


def evaluate(grid, X_val, y_val, X_test, y_test):
    # Evaluation on validation set
    y_val_pred = grid.best_estimator_.predict(X_val)
    print(" Validation Set Performance:")
    print(classification_report(y_val, y_val_pred, target_names=label_mapping.values()))

    # Confusion matrix for validation
    cm_val = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_val,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())],
        yticklabels=[label_mapping[i] for i in sorted(label_mapping.keys())],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Validation Set")
    plt.tight_layout()
    # plt.show()
    plt.savefig("metrics/tfidf_validation_confusion_matrix.png")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(
        "\nClassification Report:\n",
        classification_report(y_test, y_pred, target_names=label_mapping.values()),
    )


# # Demo: Predict sentiment of a new statement
# sample_text = "I feel so lost and disconnected from everyone around me."
# cleaned_text = clean_text(sample_text)  # Use the same cleaning function
# sample_tfidf = grid.best_estimator_.named_steps["tfidf"].transform([cleaned_text])
# predicted_label = grid.best_estimator_.named_steps["clf"].predict(sample_tfidf)[0]

# # Convert label back to string
# label_name = label_mapping[predicted_label]
# print(f" Input: {sample_text}")
# print(f" Predicted Sentiment: {label_name}")

# utils.preprocessor.preprocess_text


def demo(test_texts=None):
    return


if __name__ == "__main__":
    grid = train_tfidf_model(X_train, y_train)
    evaluate(grid, X_val, y_val, X_test, y_test)
