"""
train_email_model.py
====================
Trains the REAL machine-learning email detector and saves `email_model.pkl`.

WHY THIS SCRIPT DOES NOT RUN OUT-OF-THE-BOX
-------------------------------------------
This project does not contain a labeled email dataset (the only dataset in
`dataset/` is the PhiUSIIL *URL* dataset, which cannot classify emails).
Supply your own CSV and run this script.

REQUIRED CSV FORMAT  (default path: dataset/email_dataset.csv)
--------------------------------------------------------------
    text,label
    "Subject: Verify your account ... Dear customer ...",phishing
    "Hi team, the meeting is moved to 3pm",safe
    "WIN a FREE prize now!!! click here",spam

  * text  : the email content (sender + subject + body in one string is fine)
  * label : one of   safe | spam | phishing
            (aliases accepted: ham/legitimate/normal -> safe, phish -> phishing)

THREE-CLASS REQUIREMENT
-----------------------
The detector is designed for 3 classes. Many public datasets are only
spam-vs-ham. A spam/ham dataset CANNOT teach the model what phishing looks
like, so by default this script refuses to train unless all three labels
exist. If you really want a 2-class model, add --allow-two-class (the app
will then only ever predict the classes present in your data).

Suggested public sources (you must download them yourself and merge/convert
them into the CSV format above):
  * Phishing emails: e.g. the Nazario phishing corpus, "Phishing Email Dataset" on Kaggle
  * Spam / ham:      e.g. SpamAssassin public corpus, Enron-Spam, Kaggle "Spam Email" sets

USAGE (Windows CMD)
-------------------
    py -3.11 train_email_model.py
    py -3.11 train_email_model.py --data C:\\path\\to\\my_emails.csv
    py -3.11 train_email_model.py --data my.csv --text-col body --label-col category
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Re-use the SAME cleaning function the app uses at prediction time.
from email_analyzer import normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "dataset", "email_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "email_model.pkl")
METRICS_OUT = os.path.join(BASE_DIR, "email_model_metrics.json")

LABEL_ALIASES = {
    "safe": "safe", "ham": "safe", "legitimate": "safe", "legit": "safe", "normal": "safe",
    "spam": "spam", "junk": "spam",
    "phishing": "phishing", "phish": "phishing",
}
MIN_ROWS_PER_CLASS = 50


def main():
    ap = argparse.ArgumentParser(description="Train the PhishGuard email classifier")
    ap.add_argument("--data", default=DEFAULT_DATA, help="CSV with text,label columns")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--allow-two-class", action="store_true",
                    help="Train even if one of safe/spam/phishing is missing")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[!] Dataset not found: {args.data}\n")
        print("    A labeled email dataset is REQUIRED and is not included in this project.")
        print("    Create a CSV with two columns:  text,label   (labels: safe, spam, phishing)")
        print("    Then run:  py -3.11 train_email_model.py --data <your_file.csv>")
        sys.exit(1)

    df = pd.read_csv(args.data)
    for col in (args.text_col, args.label_col):
        if col not in df.columns:
            print(f"[!] Column '{col}' not found. Columns in file: {list(df.columns)}")
            sys.exit(1)

    df = df[[args.text_col, args.label_col]].dropna()
    df["label"] = df[args.label_col].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    unknown = df[df["label"].isna()]
    if len(unknown):
        print(f"[i] Dropping {len(unknown)} rows with unrecognised labels.")
        df = df.dropna(subset=["label"])

    counts = df["label"].value_counts().to_dict()
    print("Rows per class:", counts)

    missing = {"safe", "spam", "phishing"} - set(counts)
    if missing and not args.allow_two_class:
        print(f"\n[!] This dataset has no examples of: {sorted(missing)}")
        print("    A three-class detector (safe/spam/phishing) cannot be trained from it.")
        print("    Add the missing class, or re-run with --allow-two-class to train a model")
        print("    that only knows the classes present (it will NOT detect the missing ones).")
        sys.exit(1)
    if min(counts.values()) < MIN_ROWS_PER_CLASS:
        print(f"[!] Each class needs at least {MIN_ROWS_PER_CLASS} examples (smallest has {min(counts.values())}).")
        sys.exit(1)

    # Same cleaning as at prediction time (lower-case, URL -> 'urltoken', whitespace)
    X = df[args.text_col].astype(str).map(normalize_text)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ONE Pipeline = TF-IDF + classifier, so preprocessing can never get out of sync.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50_000,
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    print("\n", classification_report(y_test, pred, zero_division=0))
    labels = list(pipeline.classes_)
    print("Confusion matrix (rows=true, cols=predicted), order:", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    joblib.dump(pipeline, MODEL_OUT)
    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "classes": labels,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "rows_per_class": {k: int(v) for k, v in counts.items()},
            "report": report,
        }, fh, indent=2)

    print(f"\nSaved model   -> {MODEL_OUT}")
    print(f"Saved metrics -> {METRICS_OUT}")
    print("Restart the Flask app; it will now use machine learning for emails.")


if __name__ == "__main__":
    main()
