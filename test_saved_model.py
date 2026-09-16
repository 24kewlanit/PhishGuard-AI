import joblib
from feature_extractor import extract_features

# Load trained model
model = joblib.load("phishing_model.pkl")

print("======================================")
print("       PHISHING URL DETECTOR")
print("======================================")

while True:

    url = input("\nEnter URL (or type 'exit' to quit): ").strip()

    if url.lower() == "exit":
        print("\nExiting...")
        break

    if not url:
        print("Please enter a URL.")
        continue

    try:
        # Extract URL features
        features = extract_features(url)

        print("\nExtracted Features:")
        print(features.to_string(index=False))

        # Prediction
        prediction = model.predict(features)[0]

        # Probabilities
        probabilities = model.predict_proba(features)[0]

        # PhiUSIIL label mapping:
        # 0 = Phishing
        # 1 = Legitimate

        phishing_probability = probabilities[list(model.classes_).index(0)]
        legitimate_probability = probabilities[list(model.classes_).index(1)]

        print("\n======================================")
        print("URL:", url)
        print("======================================")

        print("Probability for Phishing: {:.2f} %".format(
            phishing_probability * 100
        ))

        print("Probability for Legitimate: {:.2f} %".format(
            legitimate_probability * 100
        ))

        print("--------------------------------------")

        if prediction == 0:
            print("MODEL PREDICTION: PHISHING")
            print("⚠ WARNING: This URL may be dangerous.")
        else:
            print("MODEL PREDICTION: LEGITIMATE")
            print("✓ This URL appears legitimate.")

        print("======================================")

    except Exception as e:
        print("\nError:", e)