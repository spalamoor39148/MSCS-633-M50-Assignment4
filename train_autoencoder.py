"""
Fraud Detection with PyOD AutoEncoder

Dataset expected:
    creditcard.csv

This script:
1. Loads the anonymized credit card fraud dataset
2. Splits train/test data
3. Trains PyOD AutoEncoder
4. Computes anomaly scores
5. Evaluates model performance
6. Saves outputs for submission
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from pyod.models.auto_encoder import AutoEncoder


def main() -> None:
    # -----------------------------
    # Paths
    # -----------------------------
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir /  "creditcard.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Download creditcard.csv from Kaggle and place it in the data folder."
        )

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df["Class"].value_counts())

    # Features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Optional but often useful for anomaly detection:
    # Train only on normal transactions to help the autoencoder
    X_train_normal = X_train[y_train == 0]

    # -----------------------------
    # Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Model configuration
    # contamination roughly reflects fraud ratio in dataset
    # -----------------------------
    contamination_rate = y.mean()

    model = AutoEncoder(
        contamination=contamination_rate,
        epoch_num=20,
        batch_size=64,
        hidden_neuron_list=[32, 16, 16, 32],
        preprocessing=False,
        verbose=1,
        random_state=42
    )

    # -----------------------------
    # Train model
    # -----------------------------
    model.fit(X_train_scaled)

    # -----------------------------
    # Predict on test set
    # PyOD:
    #   predict() -> binary labels
    #   decision_function() -> anomaly scores
    # -----------------------------
    y_pred = model.predict(X_test_scaled)
    y_scores = model.decision_function(X_test_scaled)

    # -----------------------------
    # Evaluation
    # -----------------------------
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\nROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # -----------------------------
    # Save model, scaler, and outputs
    # -----------------------------
    joblib.dump(model, output_dir / "ae_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")

    scores_df = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "pred_label": y_pred,
        "anomaly_score": y_scores
    })
    scores_df.to_csv(output_dir / "reconstruction_scores.csv", index=False)

    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Fraud ratio: {y.mean():.6f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()