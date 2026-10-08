from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset shape:", X.shape)
print("Number of features:", len(feature_names))
print("Classes:", data.target_names)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Build model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])


# Train
model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=data.target_names
    )
)


# Save model
os.makedirs("app", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": feature_names.tolist(),
        "target_names": data.target_names.tolist()
    },
    "app/model.pkl"
)

print("\nModel saved to app/model.pkl")