import pandas as pd
import joblib

from feature_extractor import extract_features


# ==========================================
# LOAD DATASET AND MODEL
# ==========================================

df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
model = joblib.load("phishing_model.pkl")


# ==========================================
# SELECT TEST URLs
# ==========================================

test_urls = df.sample(10, random_state=42)


print("\n======================================")
print("       DATASET URL MODEL TEST")
print("======================================\n")


# ==========================================
# TEST EACH URL
# ==========================================

correct = 0

for _, row in test_urls.iterrows():

    url = row["URL"]
    actual_label = row["label"]

    # Extract features
    features = extract_features(url)

    # Prediction
    prediction = model.predict(features)[0]

    # Probability
    probabilities = model.predict_proba(features)[0]

    phishing_probability = probabilities[1] * 100
    legitimate_probability = probabilities[0] * 100

    # Check result
    if prediction == actual_label:
        result = "CORRECT"
        correct += 1
    else:
        result = "WRONG"

    print("--------------------------------------")
    print("URL:", url)

    print("Actual Label:", actual_label)
    print("Model Prediction:", prediction)

    print(
        "Probability for Phishing:",
        round(phishing_probability, 2),
        "%"
    )

    print(
        "Probability for Legitimate:",
        round(legitimate_probability, 2),
        "%"
    )

    print("Result:", result)


# ==========================================
# FINAL RESULT
# ==========================================

print("\n======================================")
print("          TEST SUMMARY")
print("======================================")

print("URLs Tested:", len(test_urls))
print("Correct Predictions:", correct)

accuracy = (correct / len(test_urls)) * 100

print("Test Accuracy:", round(accuracy, 2), "%")

print("======================================")