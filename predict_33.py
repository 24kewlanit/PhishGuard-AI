import joblib
import pandas as pd

from feature_extractor import extract_features


print("\n========================================")
print("     33-FEATURE PHISHING DETECTION")
print("========================================")

url = input("\nEnter website URL: ").strip()

# Extract all 33 features
X = extract_features(url)

# Load trained model
model = joblib.load("phishing_model.pkl")

# Get the exact feature order expected by the model
if hasattr(model, "feature_names_in_"):

    feature_order = list(model.feature_names_in_)

    X = X[feature_order]

else:

    print("\nERROR: Model does not contain feature names.")
    exit()

# Prediction
prediction = model.predict(X)[0]

probability = model.predict_proba(X)[0]

# Your dataset uses:
# 0 = phishing
# 1 = legitimate

if prediction == 0:

    result = "PHISHING"

    confidence = probability[0] * 100

else:

    result = "LEGITIMATE"

    confidence = probability[1] * 100


print("\nURL:", url)

print("Prediction:", result)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("\nExtracted 33 Features:")

print(X.to_string(index=False))

print("\n========================================")