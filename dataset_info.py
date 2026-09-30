import pandas as pd

# Load your converted CSV
df = pd.read_csv("E:/phishing+websites/phishing_dataset.csv")

# Display the number of rows and columns
print("Shape:", df.shape)

# Show column names
print("\n🔹 Columns in dataset:")
print(df.columns.tolist())

# Show first 5 rows to understand data
print("\n🔹 Sample rows:")
print(df.head())

# Check the label distribution (phishing vs legitimate)
print("\n🔹 Label Distribution:")
print(df.iloc[:, -1].value_counts())
