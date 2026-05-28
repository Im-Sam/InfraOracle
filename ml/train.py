import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json

from preprocess import preprocess
from model import LSTMAutoencoder

# =========================
# Config
# =========================

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

PATIENCE = 2
MIN_DELTA = 1e-5

MODEL_SAVE_PATH = "../models/lstm_autoencoder.pt"

# =========================
# Load Data
# =========================

sequences = preprocess(
    "../data/demo/demo_vm_metrics.csv"
)

tensor_x = torch.tensor(
    sequences,
    dtype=torch.float32
)

dataset = TensorDataset(tensor_x)

# 80/20 train-validation split
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# Model
# =========================

model = LSTMAutoencoder(n_features=4)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# Early Stopping Variables
# =========================

best_val_loss = float("inf")

patience_counter = 0

# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

    model.train()

    train_loss = 0

    for batch in train_loader:

        x = batch[0]

        optimizer.zero_grad()

        reconstructed = model(x)

        loss = criterion(
            reconstructed,
            x
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            x = batch[0]

            reconstructed = model(x)

            loss = criterion(
                reconstructed,
                x
            )

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f}"
    )

    # ---------------------
    # Early Stopping Check
    # ---------------------

    if avg_val_loss < (best_val_loss - MIN_DELTA):

        best_val_loss = avg_val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("Best model saved.")

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{PATIENCE}"
        )

    if patience_counter >= PATIENCE:

        print("Early stopping triggered.")

        break

# =========================
# Load Best Model
# =========================

model.load_state_dict(
    torch.load(MODEL_SAVE_PATH)
)

# =========================
# Threshold Calculation
# =========================

model.eval()

with torch.no_grad():

    reconstructed = model(tensor_x)

    mse = torch.mean(
        (reconstructed - tensor_x) ** 2,
        dim=(1, 2)
    )

threshold = np.percentile(
    mse.numpy(),
    99
)

with open("../models/threshold.json", "w") as f:

    json.dump(
        {
            "threshold": float(threshold)
        },
        f
    )

print(f"Threshold saved: {threshold}")