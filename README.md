# Fraud Detection using AutoEncoder 

##  Overview

This project implements a Fraud Detection System using an AutoEncoder model from PyOD, a Python library for outlier detection. The model is trained on an anonymized credit card transactions dataset to identify fraudulent activities as anomalies.

Fraud detection is treated as an unsupervised anomaly detection problem, where fraudulent transactions are rare.

---

##  Objectives

- Build a fraud detection model using deep learning (AutoEncoder)
- Detect anomalies using reconstruction error
- Evaluate performance on imbalanced data
- Follow machine learning best practices

---



##  Dataset


Dataset: Credit Card Fraud Detection (Kaggle)

Total Transactions: 284,807

Fraud Cases: 492

Highly imbalanced dataset

Download:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place file as:
creditcard.csv

---

##  Installation
1. Create Virtual Environment
   
   python3 -m venv .venv
   
   source .venv/bin/activate

2. Install Dependencies
   
   pip install -r requirements.txt

---

##  Run

python train_autoencoder.py

---
## Methodology
### AutoEncoder (PyOD)

The AutoEncoder is a neural network that learns to reconstruct input data.

1. Trained on normal transactions
2. Learns patterns of legitimate behavior
3. Fraudulent transactions produce high reconstruction error
   
### Steps
1. Load dataset
2. Preprocess and scale features
3. Train AutoEncoder model
4. Compute anomaly scores
5. Classify transactions
6. Evaluate performance
---
## Evaluation Metrics

Due to class imbalance, we use:

1. ROC-AUC → measures ranking quality
2. PR-AUC → better for imbalanced datasets
3. Confusion Matrix
4. Precision / Recall / F1-score

---
##  Sample Output

Dataset shape: (284807, 31)

ROC-AUC: 0.95
PR-AUC: 0.1955

---

##  Outputs

- metrics.txt
- reconstruction_scores.csv
- ae_model.joblib
- scaler.joblib

---
<img width="2066" height="828" alt="image" src="https://github.com/user-attachments/assets/a5919122-c2ec-4193-8ed2-4af0c20e5e89" />


## Challenges
1. Handling highly imbalanced dataset
2. Selecting proper evaluation metrics
3. Managing dependencies like PyTorch
4. Understanding anomaly detection vs classification

---

## Key Learnings
1. AutoEncoders are effective for unsupervised fraud detection
2. Accuracy alone is misleading in imbalanced datasets
3. PR-AUC is more informative than accuracy
4. Real-world systems require careful evaluation

---

## Future Improvements
1. Hyperparameter tuning
2. Ensemble anomaly detection methods
3. Real-time fraud detection pipeline
4. Deployment using APIs

---


##  Author

Sri Sai Palamoor
