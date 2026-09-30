# model_catboost.py
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# 1️⃣ Load Data

X_train = pd.read_csv("train_features.csv")
X_test = pd.read_csv("test_features.csv")
y_train = pd.read_csv("train_labels.csv").values.ravel()
y_test = pd.read_csv("test_labels.csv").values.ravel()


# 2️⃣ Train CatBoost

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)


# 3️⃣ Evaluate

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n🔍 CatBoost Evaluation:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 4️⃣ Save Model

model.save_model("phishing_catboost_model.cbm")
print("\n✅ Model saved as phishing_catboost_model.cbm")


# 🔍 CatBoost Evaluation:
# Accuracy:  0.9756
# Precision: 0.9674
# Recall:    0.9894
# F1-score:  0.9783
# AUC:       0.9977

# Confusion Matrix:
#  [[ 939   41]
#  [  13 1218]]
