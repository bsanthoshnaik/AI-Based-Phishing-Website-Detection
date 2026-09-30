
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# 1️⃣ Load Data

X_train = pd.read_csv("train_features.csv")
X_test = pd.read_csv("test_features.csv")
y_train = pd.read_csv("train_labels.csv").values.ravel()
y_test = pd.read_csv("test_labels.csv").values.ravel()


# 2️⃣ Train Autoencoder ONLY on Legitimate URLs (y == 1)

X_train_legit = X_train[y_train == 1]
print("Training samples (legit only):", X_train_legit.shape)

input_dim = X_train.shape[1]
encoding_dim = 16  # bottleneck

# Encoder–Decoder architecture
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)


# 3️⃣ Train Model

history = autoencoder.fit(
    X_train_legit, X_train_legit,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stop],
    verbose=1
)


# 4️⃣ Compute Reconstruction Error

reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

# Threshold selection – 95th percentile of legit reconstruction error
threshold = np.percentile(mse[y_test == 1], 95)
print(f"🔹 Reconstruction Error Threshold: {threshold:.5f}")

# Predict anomalies
y_pred = (mse > threshold).astype(int)  # 1=phishing, 0=legit


# 5️⃣ Evaluate

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, mse)

print("\n🔍 Autoencoder Evaluation:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 6️⃣ Save Model

autoencoder.save("phishing_autoencoder.keras")
print("\n✅ Autoencoder saved as phishing_autoencoder.keras")

# 🔍 Autoencoder Evaluation:
# Accuracy:  0.8410
# Precision: 0.1771
# Recall:    0.0504
# F1-score:  0.0784
# AUC:       0.2524

# Confusion Matrix:
#  [[ 692  288]
#  [1169   62]]