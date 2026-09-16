import joblib
from feature_extractor import extract_features

model = joblib.load("phishing_model.pkl")

url = input("Enter website URL: ").strip()
X = extract_features(url)

prediction = int(model.predict(X)[0])
prob = model.predict_proba(X)[0]
classes = list(model.classes_)

phishing = prob[classes.index(0)] * 100
legitimate = prob[classes.index(1)] * 100

print("\nURL:", url)
print("Prediction:", "PHISHING" if prediction == 0 else "LEGITIMATE")
print(f"Phishing probability: {phishing:.2f}%")
print(f"Legitimate probability: {legitimate:.2f}%")
print("\nExtracted Features:")
print(X.to_string(index=False))
