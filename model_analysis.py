import joblib
import pandas as pd

# Load trained model
model = joblib.load("phishing_model.pkl")

# Load feature names
features = joblib.load("features.pkl")

# Get feature importance
importance = model.feature_importances_

# Create table
result = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from highest to lowest
result = result.sort_values(
    by="Importance",
    ascending=False
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)

print(result.to_string(index=False))