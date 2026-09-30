
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load Preprocessed Data

X_train = pd.read_csv("train_features.csv")
X_test = pd.read_csv("test_features.csv")
y_train = pd.read_csv("train_labels.csv").values.ravel()
y_test = pd.read_csv("test_labels.csv").values.ravel()

#  Build Model

model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Train Model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


#  Evaluate

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n🔍 Model Evaluation:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model

model.save("phishing_dnn_model.h5")
print("\n✅ Model saved as phishing_dnn_model.h5")

# 🔍 Model Evaluation:
# Accuracy:  0.9733
# Precision: 0.9658
# Recall:    0.9870
# F1-score:  0.9763
# AUC:       0.9973

# Confusion Matrix:
#  [[ 937   43]
#  [  16 1215]]