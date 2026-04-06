# 💳 Fraud Detection using AutoEncoder (PyOD)

## 📌 Overview

This project implements a Fraud Detection System using an AutoEncoder model from PyOD, a Python library for outlier detection. The model is trained on an anonymized credit card transactions dataset to identify fraudulent activities as anomalies.

Fraud detection is treated as an unsupervised anomaly detection problem, where fraudulent transactions are rare.

---

## 🎯 Objectives

- Build a fraud detection model using deep learning (AutoEncoder)
- Detect anomalies using reconstruction error
- Evaluate performance on imbalanced data
- Follow machine learning best practices

---

## 📂 Project Structure

fraud-detection-autoencoder/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   └── train_autoencoder.py
│
├── outputs/
│   ├── metrics.txt
│   ├── reconstruction_scores.csv
│   ├── ae_model.joblib
│   └── scaler.joblib
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 📊 Dataset

Dataset: Credit Card Fraud Detection (Kaggle)

Download:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place file in:
data/creditcard.csv

---

## ⚙️ Installation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## 🚀 Run

python src/train_autoencoder.py

---

## 📈 Sample Output

Dataset shape: (284807, 31)

ROC-AUC: 0.95
PR-AUC: 0.1955

---

## 📁 Outputs

- metrics.txt
- reconstruction_scores.csv
- ae_model.joblib
- scaler.joblib

---

## 🧑‍💻 Author

Sri Sai Palamoor
