# InfraOracle

An AI-powered predictive monitoring system for virtual machines using LSTM-based anomaly detection.

It simulates SRE-grade observability by forecasting infrastructure failures before they occur.

---

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange)
![Status](https://img.shields.io/badge/status-active-success)

## 🚀 Features

- LSTM Autoencoder for anomaly detection
- Per-VM behavioral baselines
- Dynamic thresholding per machine
- Root cause explanation engine
- Failure risk scoring (0–100)
- Fleet-level monitoring (50+ VMs)
- Interactive Dash observability dashboard

---

## 🧠 Architecture

Data → Feature Engineering → LSTM Autoencoder → Reconstruction Error → Anomaly Detection → Explanation Layer → Dashboard

---

## 📊 VM Simulation

The system includes a realistic infrastructure simulator with:

- 50 virtual machines
- Role-based behavior:
  - Web servers
  - API gateways
  - Databases
  - Cache nodes
  - Batch workers
- Failure drift modeling
- Traffic bursts and load spikes

---

## 🖥️ Dashboard

The UI includes:

- CPU / Memory / Disk / Network charts
- Per-VM drill-down view
- Fleet-level summary
- Failure risk scoring
- Explainable anomaly detection

---

## Setup

### Create venv

python -m venv venv

### Activate

Windows:
venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

### Install

pip install -r requirements.txt

### Generate Demo Data

python scripts/generate_demo_data.py

### Train Model

cd ml
python train.py

### Run Dashboard

cd ../
python -m app.dashboard.py