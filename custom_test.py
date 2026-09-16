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
        "IsDomainIP": 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain) else 0,
        "NoOfSubDomain": max(len(domain.split(".")) - 2, 0),
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(
            not c.isalnum() and c not in "/:.-_?=&" for c in url
        ),
        "IsHTTPS": 1
    }

    return pd.DataFrame([data])[features]

urls = [
    "https://www.aap.org",
    "https://www.saffronart.com",
    "https://43.134.167.94/servicelogin?continue=https://accounts.google.com"
]

for url in urls:
    X = extract_features(url)
    pred = model.predict(X)[0]

    if pred == 1:
        print("\nLEGITIMATE :", url)
    else:
        print("\nPHISHING    :", url)