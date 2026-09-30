import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# === 1. Load your test data ===
X_test = pd.read_csv("test_features.csv")
y_test = pd.read_csv("test_labels.csv").values.ravel()

# === 2. Load your trained models ===
from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier

# Load DNN model
dnn_model = load_model("phishing_dnn_model.h5")

# Load CatBoost model
catboost_model = CatBoostClassifier()
catboost_model.load_model("phishing_catboost_model.cbm")

# Load Autoencoder model
autoencoder = load_model("phishing_autoencoder.keras")

# === 3. Get predicted probabilities ===
# DNN and CatBoost provide probabilities directly
y_pred_dnn = dnn_model.predict(X_test)
y_pred_cat = catboost_model.predict_proba(X_test)[:, 1]

# For Autoencoder: reconstruction error as anomaly score
reconstructed = autoencoder.predict(X_test)
reconstruction_error = ((X_test - reconstructed) ** 2).mean(axis=1)
# Normalize to 0–1
y_pred_auto = (reconstruction_error - reconstruction_error.min()) / (
    reconstruction_error.max() - reconstruction_error.min()
)

# === 4. Organize predictions ===
models = {
    "Autoencoder": y_pred_auto,
    "CatBoost": y_pred_cat,
    "DNN": y_pred_dnn.ravel(),
}

# === 5. Plot ROC Curves ===
plt.figure(figsize=(8, 6))
for name, y_pred in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Different Models")
plt.legend(loc="lower right")
plt.show()

# === 6. Plot Precision–Recall Curves ===
plt.figure(figsize=(8, 6))
for name, y_pred in models.items():
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    plt.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves for Different Models")
plt.legend(loc="lower left")
plt.show()
