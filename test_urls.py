import joblib
import pandas as pd
from urllib.parse import urlparse
import re

model = joblib.load("phishing_model.pkl")
features = joblib.load("features.pkl")


def extract_features(url):

    parsed = urlparse(url)
    domain = parsed.netloc

    data = {
        "URLLength": len(url) - 1,
        "DomainLength": len(domain),
        "IsDomainIP": 1 if re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}",
            domain
        ) else 0,
        "NoOfSubDomain": max(len(domain.split(".")) - 2, 0),
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(
            not c.isalnum() and c not in "/:.-_?=&"
            for c in url
        ),
        "IsHTTPS": 1
    }

    return pd.DataFrame([data])[features]


test_urls = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://www.microsoft.com"
]


print("\nPHISHING WEBSITE DETECTION")
print("=" * 50)

for url in test_urls:

    X = extract_features(url)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    if prediction == 1:
        result = "LEGITIMATE"
        confidence = probability[1] * 100
    else:
        result = "PHISHING"
        confidence = probability[0] * 100

    print("\nURL:", url)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")

print("\n" + "=" * 50)