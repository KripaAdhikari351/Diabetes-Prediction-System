"""
train_model.py

Trains a logistic regression classifier to predict diabetes risk from
patient health metrics, then saves the fitted model and feature scaler
to disk so app.py can load them for live predictions.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("diabetes_dataset.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. Feature scaling
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Train logistic regression
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("=" * 45)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("=" * 45)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 6. Save model + scaler + the feature column order
# ---------------------------------------------------------
joblib.dump(model, "logistic_regression_model.joblib")
joblib.dump(scaler, "diabetes_scaler.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")

print("\nSaved: logistic_regression_model.joblib, diabetes_scaler.joblib, feature_columns.joblib")
