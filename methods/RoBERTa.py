import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_scheduler,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from collections import Counter
import os
import utils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        if hasattr(labels, "values"):
            self.labels = torch.tensor(labels.values, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    return SentimentDataset(encodings, labels)


def train_roberta_model(checkpoint_path="saved_models/roberta_model.pth"):
    """Train RoBERTa model with checkpoint functionality"""
    # Check if checkpoint already exists
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        model, tokenizer, label_encoder = load_model(checkpoint_path)
        return model, tokenizer, label_encoder

    print("Training new RoBERTa model...")

    # Load and preprocess data
    df = utils.load_data()

    # Data filtering
    print(f"Dataset size before dropping missing items: {len(df)}")
    df = df.dropna(subset=["statement", "status"])
    df = df[~df["statement"].str.strip().eq("")]
    df = df[~df["status"].str.strip().eq("")]
    print(f"Dataset size after dropping missing items: {len(df)}")

    # Encoding labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["status"])

    # Data split (70% train, 20% val, 10% test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["statement"],
        df["label"],
        stratify=df["label"],
        test_size=0.30,
        random_state=42,
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, stratify=temp_labels, test_size=0.333, random_state=42
    )

    print(
        f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}"
    )

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(label_encoder.classes_)
    )
    model.to(device)

    # Prepare datasets
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_data(val_texts, val_labels, tokenizer)
    test_dataset = tokenize_data(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Setup training components
    optimizer = AdamW(model.parameters(), lr=2e-5)
    number_epochs = 3
    num_training_steps = number_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Class weights for balanced training
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(number_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_function(outputs.logits, batch["labels"])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(
            f"Epoch {epoch+1}/{number_epochs} - Loss: {total_loss/len(train_loader):.4f}"
        )

    # Save model
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "model_config": {
            "num_labels": len(label_encoder.classes_),
            "model_name": "roberta-base",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    return model, tokenizer, label_encoder


def evaluate_roberta_model(model, tokenizer, label_encoder):
    """Evaluate RoBERTa model on test data"""
    # Reload data for evaluation
    df = utils.load_data()
    df = df.dropna(subset=["statement", "status"])
    df = df[~df["statement"].str.strip().eq("")]
    df = df[~df["status"].str.strip().eq("")]

    # Use same label encoding
    df["label"] = label_encoder.transform(df["status"])

    # Same data split as training
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["statement"],
        df["label"],
        stratify=df["label"],
        test_size=0.30,
        random_state=42,
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, stratify=temp_labels, test_size=0.333, random_state=42
    )

    # Prepare test dataset
    test_dataset = tokenize_data(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Generate and save classification report
    report = classification_report(
        all_labels, all_preds, target_names=label_encoder.classes_
    )
    print("\nTest Classification Report:")
    print(report)

    # Save classification report to file
    os.makedirs("metrics/roberta", exist_ok=True)
    with open("metrics/roberta/roberta_classification_report.txt", "w") as f:
        f.write("RoBERTa Test Classification Report\n")
        f.write("=" * 40 + "\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("RoBERTa Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("metrics/roberta/roberta_test_confusion_matrix.png")
    plt.close()

    print(
        "Classification report saved to metrics/roberta/roberta_classification_report.txt"
    )
    print("Confusion matrix saved to metrics/roberta/roberta_test_confusion_matrix.png")


def load_model(checkpoint_path="saved_models/roberta_model.pth"):
    """Load trained RoBERTa model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load tokenizer and label encoder
    tokenizer = checkpoint["tokenizer"]
    label_encoder = checkpoint["label_encoder"]

    # Reconstruct model
    model = RobertaForSequenceClassification.from_pretrained(
        checkpoint["model_config"]["model_name"],
        num_labels=checkpoint["model_config"]["num_labels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, tokenizer, label_encoder


def predict_sentiment(text, model, tokenizer, label_encoder):
    """Predict sentiment for a single text"""
    model.eval()

    # Tokenize input
    dummy_dataset = tokenize_data([text], [0], tokenizer)  # [0] = dummy label
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1)

    with torch.no_grad():
        for batch in dummy_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            # Get prediction and probabilities
            pred_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_label = label_encoder.classes_[pred_class]
    return predicted_label, probabilities


def demo(test_texts=None, expected_labels=None):
    """Demo function using saved RoBERTa model - returns results instead of printing"""
    try:
        model, tokenizer, label_encoder = load_model()
    except:
        return None

    # if not provided, use some default texts with expected labels
    if not test_texts:
        test_texts = [
            "I feel restless",  # anxiety
            "Beautiful morning",  # normal
            "I did not ask to be born",  # depression
            "This exam is stressing me out",  # stress
        ]
        expected_labels = ["Anxiety", "Normal", "Depression", "Stress"]

    results = []
    for i, text in enumerate(test_texts):
        prediction, probs = predict_sentiment(text, model, tokenizer, label_encoder)

        # Create probability dictionary and sort by value
        prob_dict = dict(zip(label_encoder.classes_, probs))
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        # Determine if prediction is correct
        expected = (
            expected_labels[i] if expected_labels and i < len(expected_labels) else None
        )
        is_correct = prediction == expected if expected else None

        results.append(
            {
                "text": text,
                "prediction": prediction,
                "expected": expected,
                "correct": is_correct,
                "probabilities": sorted_probs,
            }
        )

    return results


if __name__ == "__main__":
    # Train model
    model, tokenizer, label_encoder = train_roberta_model()

    # Evaluate model
    evaluate_roberta_model(model, tokenizer, label_encoder)

    # Run demo
    # demo()
