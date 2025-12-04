import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# MODEL
# -------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(64, 128, 4), nn.ReLU(), nn.BatchNorm2d(128),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# DATASET
# -------------------------
class DigitDataset(Dataset):
    def __init__(self, df, has_labels=True):
        self.has_labels = has_labels

        if has_labels:
            self.y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long).to(DEVICE)
            self.X = df.iloc[:, 1:].values
        else:
            self.X = df.values

        self.X = (
            torch.tensor(self.X, dtype=torch.float32)
            .reshape(-1, 1, 28, 28)
            .to(DEVICE) / 255.0
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        total_train_loss = 0

        # -----------------------
        # TRAINING LOOP
        # -----------------------
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # compute train accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        train_acc = correct_train / total_train
        train_loss = total_train_loss / len(train_loader)

        # -----------------------
        # VALIDATION LOOP
        # -----------------------
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss_sum = 0

        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X)
                loss = criterion(preds, y)
                val_loss_sum += loss.item()

                _, predicted = torch.max(preds, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

        val_acc = correct_val / total_val
        val_loss = val_loss_sum / len(val_loader)

        # -----------------------
        # PRINT LIKE KERAS
        # -----------------------
        print(
            f"Epoch {epoch+1}/{epochs} "
            f" - loss: {train_loss:.4f} - acc: {train_acc:.4f} "
            f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

    return model

# -------------------------
# ENSEMBLE TRAINING
# -------------------------
def train_ensemble():
    train_df = pd.read_csv("Data/train.csv")
    dataset = DigitDataset(train_df)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    for i in range(5):
        print(f"\n====== Training Model {i+1} ======")
        model = DigitCNN().to(DEVICE)
        model = train_model(model, train_loader, val_loader)

        torch.save(model.state_dict(), f"digit_cnn_{i}.pth")
        print(f"Saved: digit_cnn_{i}.pth")


# -------------------------
# ENSEMBLE PREDICTION
# -------------------------
def predict_ensemble():
    test_df = pd.read_csv("Data/test.csv")
    test_dataset = DigitDataset(test_df, has_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=128)

    models = []
    for i in range(5):
        model = DigitCNN().to(DEVICE)
        model.load_state_dict(torch.load(f"digit_cnn_{i}.pth"))
        model.eval()
        models.append(model)

    predictions = []
    with torch.no_grad():
        for X in tqdm(test_loader):
            logits = None
            for m in models:
                out = m(X)
                logits = out if logits is None else logits + out
            _, pred = torch.max(logits, 1)
            predictions.extend(pred.cpu().numpy())

    pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions
    }).to_csv("submission.csv", index=False)

    print("Saved submission.csv")


# -------------------------
# LOAD ENSEMBLE FOR DRAWING
# -------------------------
def load_ensemble():
    models = []
    for i in range(5):
        model = DigitCNN().to(DEVICE)
        model.load_state_dict(torch.load(f"digit_cnn_{i}.pth"))
        model.eval()
        models.append(model)
    return models


def predict_digit_from_image(img, models):
    img = torch.tensor(img, dtype=torch.float32).reshape(1, 1, 28, 28) / 255.0

    with torch.no_grad():
        logits = None
        for m in models:
            out = m(img)
            logits = out if logits is None else logits + out
        pred = torch.argmax(logits, dim=1).item()

    return pred

if __name__ == "__main__":
    predict_ensemble()
