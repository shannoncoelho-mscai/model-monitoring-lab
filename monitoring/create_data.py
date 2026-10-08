import pandas as pd

from sklearn.datasets import load_breast_cancer


# Load dataset
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)


# Create reference data
reference_data = df.sample(
    n=300,
    random_state=42
).reset_index(drop=True)


# Create current data
current_data = df.sample(
    n=300,
    random_state=7
).reset_index(drop=True)


# Introduce controlled data drift
drift_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

for feature in drift_features:
    current_data[feature] = current_data[feature] * 1.20


# Save monitoring datasets
reference_data.to_csv(
    "data/reference_data.csv",
    index=False
)

current_data.to_csv(
    "data/current_data.csv",
    index=False
)


print("Reference data shape:", reference_data.shape)
print("Current data shape:", current_data.shape)

print("\nDrift introduced in:")
for feature in drift_features:
    print("-", feature)

print("\nReference data saved to data/reference_data.csv")
print("Current data saved to data/current_data.csv")