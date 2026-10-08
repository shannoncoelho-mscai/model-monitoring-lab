import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report


# Load trained model
model_data = joblib.load("app/model.pkl")

model = model_data["model"]
target_names = model_data["target_names"]


# Load original dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Create current data using the same sampling method
current_indices = pd.Series(range(len(X))).sample(
    n=300,
    random_state=7
).values

X_current = X.iloc[current_indices].copy()
y_current = y[current_indices]


# Apply the same controlled drift
drift_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

for feature in drift_features:
    X_current[feature] = X_current[feature] * 1.20


# Generate predictions
y_pred = model.predict(X_current)


# Evaluate model
accuracy = accuracy_score(y_current, y_pred)

print("Current Data Model Evaluation")
print("=" * 60)

print(f"Current data samples: {len(X_current)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_current,
        y_pred,
        target_names=target_names
    )
)