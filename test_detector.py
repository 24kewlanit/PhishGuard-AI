import sys
import joblib
from feature_extractor import extract_features

TEST_URLS = [
    ("LEGITIMATE", "https://www.google.com"),
    ("LEGITIMATE", "https://www.microsoft.com"),
    ("LEGITIMATE", "https://www.wikipedia.org"),
    ("LEGITIMATE", "https://www.sbi.co.in"),
    ("PHISHING", "https://google.com.evil-example.com/login"),
    ("PHISHING", "https://secure-google-login-example.com/verify"),
    ("PHISHING", "http://192.168.1.10/login/verify"),
]

model = joblib.load("phishing_model.pkl")

print("\nPHISHGUARD MODEL SANITY TEST")
print("-" * 70)

failed = 0
for expected, url in TEST_URLS:
    X = extract_features(url)
    X = X[list(model.feature_names_in_)]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    phishing = float(proba[list(model.classes_).index(0)]) * 100
    result = "PHISHING" if pred == 0 else "LEGITIMATE"
    ok = result == expected
    print(f"{'PASS' if ok else 'FAIL':4} | expected={expected:10} got={result:10} | {phishing:6.2f}% phishing | {url}")
    failed += int(not ok)

print("-" * 70)
print("All sanity cases passed." if failed == 0 else f"{failed} sanity case(s) failed.")
sys.exit(1 if failed else 0)
