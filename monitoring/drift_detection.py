import pandas as pd
from scipy.stats import ks_2samp


# Load reference and current data
reference_data = pd.read_csv("data/reference_data.csv")
current_data = pd.read_csv("data/current_data.csv")


# Significance level
alpha = 0.05


print("Kolmogorov-Smirnov Drift Detection")
print("=" * 60)

drift_count = 0

results = []

for feature in reference_data.columns:

    reference_values = reference_data[feature]
    current_values = current_data[feature]

    statistic, p_value = ks_2samp(
        reference_values,
        current_values
    )

    drift_detected = p_value < alpha

    if drift_detected:
        drift_count += 1

    results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })

    print(
        f"{feature}: "
        f"KS={statistic:.4f}, "
        f"p-value={p_value:.6f}, "
        f"drift={drift_detected}"
    )


print("\n" + "=" * 60)
print(f"Features tested: {len(reference_data.columns)}")
print(f"Features with detected drift: {drift_count}")
print(f"Significance level: {alpha}")

print("\nDrift Summary")
print("=" * 60)

for result in results:
    if result["drift_detected"]:
        print(
            f"{result['feature']}: "
            f"KS={result['ks_statistic']:.4f}, "
            f"p-value={result['p_value']:.6f}"
        )

print("\nConclusion:")
print(
    f"{drift_count} out of {len(reference_data.columns)} "
    "features show statistically significant drift "
    f"at alpha={alpha}."
)