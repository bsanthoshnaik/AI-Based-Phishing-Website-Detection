# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV file (update the path to your file)
df = pd.read_csv(r"E:\phishing+websites\phishing_dataset.csv")

print("✅ Dataset Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Map target values: -1 -> 0 (phishing), 1 -> 1 (legitimate)
df['Result'] = df['Result'].map({-1: 0, 1: 1})

# Separate features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Train shape:", X_train_scaled.shape)
print("✅ Test shape:", X_test_scaled.shape)

# Save processed data for later model training
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("train_features.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("test_features.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

print("🎯 Preprocessing complete and saved as CSV files.")

# https://www.kaggle.com/datasets/nitsey/dataset-phising-website?select=Dataset+Phising+Website.csv