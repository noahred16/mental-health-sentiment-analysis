import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from tqdm import tqdm
import utils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Vocabulary builder
class Vocabulary:
    def __init__(self, max_vocab_size=10000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()
        self.max_vocab_size = max_vocab_size

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
class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.5,
    ):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Only use dropout in LSTM if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
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


# Main execution
def main():
    # Load data
    df = utils.load_data()

    # TEMP Sample the data for faster testing (use 1% of data)
    SAMPLE_FRACTION = 0.0025  # Change this to 1.0 for full dataset
    if SAMPLE_FRACTION < 1.0:
        print(f"\nSampling {SAMPLE_FRACTION*100}% of data for faster testing...")
        # Stratified sampling to maintain class distribution
        df = df.groupby("status", group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_FRACTION, random_state=42)
        )
        print(f"Working with {len(df)} samples")
        print(f"Sampled class distribution:\n{df['status'].value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    df["status"] = label_encoder.fit_transform(df["status"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"].values,
        df["status"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["status"].values,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Build vocabulary
    vocab = Vocabulary(max_vocab_size=10000)  # Restored to 10000 for better coverage
    vocab.build_vocab(X_train)

    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train, vocab)
    test_dataset = MentalHealthDataset(X_test, y_test, vocab)

    # Create dataloaders
    batch_size = 8  # Restored to 64 for better GPU utilization
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model parameters
    vocab_size = len(vocab.word2idx)
    embedding_dim = 200  # Restored to 200
    hidden_dim = 256  # Restored to 256
    num_layers = 2  # Restored to 2 layers
    num_classes = len(label_encoder.classes_)
    dropout = 0.5

    # Initialize model
    model = SentimentLSTM(
        vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    # Training loop
    num_epochs = 1  # Temp 1 for debugging, Restore to 20 epochs for better convergence
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
            torch.save(model.state_dict(), "saved_models/lstm_model.pth")
            print("Saved best model!")

    # Load best model for final evaluation
    model.load_state_dict(torch.load("saved_models/lstm_model.pth"))

    # Final evaluation
    print("\nFinal evaluation on test set:")
    _, test_acc, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels, predictions, target_names=label_encoder.classes_
        )
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("metrics/lstm_confusion_matrix.png")
    plt.show()

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
    plt.savefig("metrics/lstm_training_history.png")
    plt.show()

    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")

    # Example prediction function
    def predict_sentiment(text, model, vocab, label_encoder):
        model.eval()
        processed_text = utils.preprocess_text(text)
        indices = vocab.text_to_indices(processed_text)
        indices_tensor = torch.tensor([indices]).to(device)

        with torch.no_grad():
            output = model(indices_tensor)
            _, predicted = torch.max(output, 1)

        prediction = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        return prediction, probabilities

    # Test with some examples
    test_texts = [
        "I feel anxious and can't sleep properly",
        "Life is wonderful and I'm enjoying every moment",
        "I don't see any point in continuing anymore",
        "Work pressure is getting too much for me",
    ]

    print("\nExample predictions:")
    for text in test_texts:
        prediction, probs = predict_sentiment(text, model, vocab, label_encoder)
        print(f"\nText: '{text}'")
        print(f"Predicted: {prediction}")
        print("Probabilities:")
        for cls, prob in zip(label_encoder.classes_, probs):
            print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
