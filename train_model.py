import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from feature_extractor import FEATURE_NAMES

DATASET_PATH = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH).dropna(subset=["URL", "label"]).copy()
urls = df["URL"].astype(str)

has_scheme = urls.str.contains(
    r"^[A-Za-z][A-Za-z0-9+.-]*://", regex=True, na=False
)
u = urls.where(has_scheme, "https://" + urls)

host = (
    u.str.extract(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]*)", expand=False)
    .fillna("")
    .str.split("@").str[-1].str.split(":").str[0]
)
parts = host.str.split(".")
is_ip = host.str.fullmatch(r"\d{1,3}(\.\d{1,3}){3}").fillna(False).astype(int)
url_len = u.str.len().clip(lower=1)
letters = u.str.count(r"[A-Za-z]")
digits = u.str.count(r"\d")
special = u.str.count(r"[^A-Za-z0-9]")
obfuscated = u.str.count(r"%[0-9A-Fa-f]{2}")

X = pd.DataFrame({
    "URLLength": u.str.len(),
    "DomainLength": host.str.len(),
    "IsDomainIP": is_ip,
    "NoOfSubDomain": parts.str.len().sub(2).clip(lower=0),
    "NoOfLettersInURL": letters,
    "NoOfDegitsInURL": digits,
    "NoOfEqualsInURL": u.str.count("="),
    "NoOfQMarkInURL": u.str.count(r"\?"),
    "NoOfAmpersandInURL": u.str.count("&"),
    "NoOfOtherSpecialCharsInURL": u.str.count(r"[!#$%'+,;:<>\[\]^`{|}~]"),
    "IsHTTPS": u.str.lower().str.startswith("https://").astype(int),
    "TLDLength": parts.str[-1].fillna("").str.len().where(is_ip.eq(0), 0),
    "NoOfObfuscatedChar": obfuscated,
    "ObfuscationRatio": obfuscated / url_len,
    "LetterRatioInURL": letters / url_len,
    "DegitRatioInURL": digits / url_len,
    "SpacialCharRatioInURL": special / url_len,
})[FEATURE_NAMES]

y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_phishing": float(precision_score(y_test, pred, pos_label=0)),
    "recall_phishing": float(recall_score(y_test, pred, pos_label=0)),
    "f1_phishing": float(f1_score(y_test, pred, pos_label=0)),
    "phishing_label": 0,
    "legitimate_label": 1,
    "features": FEATURE_NAMES,
}

joblib.dump(model, "phishing_model.pkl")
joblib.dump(FEATURE_NAMES, "features.pkl")

with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nMODEL TRAINED SUCCESSFULLY")
print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Phishing precision: {metrics['precision_phishing']*100:.2f}%")
print(f"Phishing recall: {metrics['recall_phishing']*100:.2f}%")
print(f"Phishing F1: {metrics['f1_phishing']*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(
    y_test, pred, target_names=["Phishing", "Legitimate"]
))
