import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean text values
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# Encode categorical columns
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

# Features and target
X = df.drop(["loan_id", "loan_status"], axis=1)
y = df["loan_status"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")