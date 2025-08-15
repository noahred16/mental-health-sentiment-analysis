import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from tqdm import tqdm
import utils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Vocabulary builder
class Vocabulary:
    def __init__(self, max_vocab_size):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()
        self.max_vocab_size = max_vocab_size
        # Hardcoded label encoder mapping
        self.label_encoder = {
            "Anxiety": 0,
            "Bipolar": 1,
            "Depression": 2,
            "Normal": 3,
            "Personality disorder": 4,
            "Stress": 5,
            "Suicidal": 6,
        }

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            words = text.split()
            self.word_freq.update(words)

        # Keep most common words
        most_common = self.word_freq.most_common(self.max_vocab_size - 2)

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def text_to_indices(self, text):
        """Convert text to indices"""
        words = text.split()
        indices = []
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx["<UNK>"])
        return indices


# Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    # using a class to represent the data allows for easy conversion to tensors!
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to indices
        indices = self.vocab.text_to_indices(text)

        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


# Custom collate function for padding
def collate_fn(batch):
    texts, labels = zip(*batch)

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return texts_padded, labels


# LSTM Model
# TODO Try Different Architecture Configurations
# larger vocabulary size
# maybe try attention mechanism
# self.attention = nn.Linear(hidden_dim * 2, 1)
# toggle bidirectional
class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.5,
        bidirectional=True,
    ):
        super(SentimentLSTM, self).__init__()

        # embedding layer ignores padding tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Only use dropout in LSTM if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state from both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Dropout and fully connected
        output = self.dropout(hidden_combined)
        output = self.fc(output)

        return output


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in tqdm(dataloader, desc="Training"):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_predictions, all_labels


def load_model(model_path="saved_models/lstm_model.pth"):
    """Load model and vocabulary from checkpoint"""
    import sys

    # Add current module to sys.modules to handle pickle loading
    sys.modules["__main__"] = sys.modules[__name__]

    # Load checkpoint with CPU mapping to handle CUDA/CPU compatibility
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    vocab = checkpoint["vocab"]
    params = checkpoint["model_params"]

    model = SentimentLSTM(
        vocab_size=params["vocab_size"],
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_classes=params["num_classes"],
        dropout=params["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab


def train(
    max_vocab_size=10000,
    embedding_dim=200,  # Restored to 200
    hidden_dim=256,  # Restored to 256
    num_layers=2,  # Restored to 2 layers
    dropout=0.5,
    model_name="regular",
    model_path="saved_models/lstm_model.pth",
    bidirectional=True,
):
    # Load data
    df = utils.load_data()

    # Build vocabulary
    vocab = Vocabulary(max_vocab_size)
    df["status"] = df["status"].map(vocab.label_encoder)

    # Split data
    X_train, X_test, y_train, y_test = utils.get_train_test_split(df)

    # Vocab built only from training set
    vocab.build_vocab(X_train)

    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train, vocab)
    test_dataset = MentalHealthDataset(X_test, y_test, vocab)

    # Create dataloaders
    batch_size = 64  # Restored to 64 for better GPU utilization
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model parameters
    vocab_size = len(vocab.word2idx)
    num_classes = len(vocab.label_encoder)

    # Initialize model
    model = SentimentLSTM(
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout,
        bidirectional,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    # Training loop
    num_epochs = 10  # Reset to 20 epochs for better convergence
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    print("\nStarting training...")
    best_test_acc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        # Scheduler step
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "model_params": {
                        "vocab_size": vocab_size,
                        "embedding_dim": embedding_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "num_classes": num_classes,
                        "dropout": dropout,
                    },
                },
                model_path,
            )
            print("Saved best model!")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics/lstm/lstm_training_history_" + model_name + ".png")
    plt.close()

    print(f"\nBest Test Accuracy from training: {best_test_acc:.4f}")


def evaluate_model(model_path, model_name="lstm"):
    """Evaluate saved model on test set"""
    model, vocab = load_model(model_path)

    # Recreate test data
    df = utils.load_data()
    df["status"] = df["status"].map(vocab.label_encoder)

    X_train, X_test, y_train, y_test = utils.get_train_test_split(df)

    test_dataset = MentalHealthDataset(X_test, y_test, vocab)
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    criterion = nn.CrossEntropyLoss()

    _, test_acc, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )

    status_labels = list(vocab.label_encoder.keys())

    # Classification report
    report = classification_report(true_labels, predictions, target_names=status_labels)
    print("\nClassification Report:")
    print(report)

    # Save classification report to file
    with open(f"metrics/lstm/lstm_classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=status_labels,
        yticklabels=status_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("metrics/lstm/lstm_test_confusion_matrix.png")
    plt.close()

    print(f"\nTest Accuracy from evaluate_model: {test_acc:.4f}")
    print("Confusion matrix saved to metrics/lstm/lstm_test_confusion_matrix.png")

    return model, vocab


def predict_sentiment(text, model, vocab):
    model.eval()
    processed_text = utils.preprocessor.preprocess_text(text)
    indices = vocab.text_to_indices(processed_text)
    indices_tensor = torch.tensor([indices]).to(device)

    with torch.no_grad():
        output = model(indices_tensor)
        _, predicted = torch.max(output, 1)

    idx_to_class = {v: k for k, v in vocab.label_encoder.items()}
    prediction = idx_to_class[predicted.cpu().numpy()[0]]
    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    return prediction, probabilities


def demo(test_texts=None, expected_labels=None):
    """Demo function using saved model - returns results instead of printing"""
    try:
        model, vocab = load_model()
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
        prediction, probs = predict_sentiment(text, model, vocab)

        # Create probability dictionary and sort by value
        prob_dict = dict(zip(vocab.label_encoder.keys(), probs))
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
    cases = [
        {
            "max_vocab_size": 2000,
            "embedding_dim": 200,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.2,
            "model_name": "lstm_model",
            "model_path": "saved_models/lstm_model.pth",
            "bidirectional": True,
        }
    ]

    # Train first
    for case in cases:
        print(f"\nTraining model with params: {case}")
        train(**case)  # dictionary unpacking, neat

    # Load and evaluate the model
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Evaluating model with params: {case}")
        print(f"{'='*60}")
        model, vocab = load_model(case["model_path"])
        evaluate_model(case["model_path"], case["model_name"])

    # demo()  # Run demo with the trained model
