import os
import torch
import json
import numpy as np
import joblib

from ml.lstm_autoencoder import LSTMAutoencoder

# =========================
# Base directory (project root)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_autoencoder.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")

# =========================
# Load model
# =========================

model = LSTMAutoencoder(n_features=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# =========================
# Load scaler + threshold
# =========================

scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["threshold"]

# =========================
# Inference function
# =========================

def predict(sequence):

    scaled = scaler.transform(sequence)
    tensor = torch.tensor([scaled], dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(tensor)

        error = (reconstructed - tensor).squeeze(0)

        mse = torch.mean(error ** 2).item()

    anomaly = mse > threshold
    failure_risk = min(100, int((mse / threshold) * 100))


    # =========================
    # FEATURE-LEVEL EXPLANATION
    # =========================

    feature_error = torch.mean(error ** 2, dim=0).numpy()

    cpu_err, mem_err, disk_err, net_err = feature_error

    explanation_parts = []

    if cpu_err > 0.02:
        explanation_parts.append("CPU spike")

    if mem_err > 0.02:
        explanation_parts.append("Memory pressure")

    if disk_err > 0.02:
        explanation_parts.append("Disk IO anomaly")

    if net_err > 0.02:
        explanation_parts.append("Network instability")

    explanation = " + ".join(explanation_parts) if explanation_parts else "Normal behavior"

    return {
        "mse": mse,
        "anomaly": anomaly,
        "failure_risk": failure_risk,
        "explanation": explanation
    }