import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

print("🚀 Training script started")

# Show current directory (debug)
print("📂 Current working directory:", os.getcwd())

# Load dataset
data_path = "data/credit_data.csv"
print("📄 Loading data from:", data_path)

data = pd.read_csv(data_path)
print("✅ Data loaded successfully")
print(data.head())

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest (black-box model)
print("🧠 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and training data
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(X_train, "model/X_train.pkl")

print("✅ Model trained and saved successfully!")
