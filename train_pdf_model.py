"""
train_pdf_model.py
==================
OPTIONAL: trains a machine-learning PDF classifier and saves `pdf_model.pkl`.

This project ships NO labeled PDF dataset, so the app uses static rules
(pdf_model.py) by default. Use this script only if you collect your own
labeled PDFs. Nothing here executes a PDF - files are only parsed statically
by pdf_analyzer.py.

TWO WAYS TO SUPPLY DATA
-----------------------
A) Folders of PDF files (the script extracts the features for you):

     py -3.11 train_pdf_model.py --safe-dir C:\\data\\clean ^
                                 --malicious-dir C:\\data\\malicious ^
                                 [--suspicious-dir C:\\data\\suspicious]

   Handle real malware samples ONLY inside an isolated virtual machine.
   Public malicious-PDF collections exist (e.g. Contagio PDF dump,
   CIC-Evasive-PDFMal2022 files); download them yourself.

B) A CSV that already contains this project's feature columns + a label:

     py -3.11 train_pdf_model.py --csv my_pdf_features.csv

   Columns required: every name in pdf_model.PDF_FEATURE_NAMES, plus `label`
   with values  safe | suspicious | malicious.
   NOTE: public CSV datasets usually use DIFFERENT feature names; they cannot
   be used directly. Re-extract features from the original PDFs (option A).

Each class needs at least 30 samples.
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from pdf_model import PDF_FEATURE_NAMES, findings_to_features
from pdf_analyzer import extract_pdf_findings, PYMUPDF_AVAILABLE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(BASE_DIR, "pdf_model.pkl")
METRICS_OUT = os.path.join(BASE_DIR, "pdf_model_metrics.json")
MIN_PER_CLASS = 30


def rows_from_folder(folder, label):
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, name)
        try:
            feats = findings_to_features(extract_pdf_findings(path))
            feats["label"] = label
            rows.append(feats)
        except Exception as exc:
            print(f"  skipped {name}: {type(exc).__name__}")
    print(f"  {label}: {len(rows)} PDFs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--safe-dir")
    ap.add_argument("--suspicious-dir")
    ap.add_argument("--malicious-dir")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [c for c in PDF_FEATURE_NAMES + ["label"] if c not in df.columns]
        if missing:
            print("[!] CSV is missing columns:", missing)
            sys.exit(1)
    elif args.safe_dir and args.malicious_dir:
        if not PYMUPDF_AVAILABLE:
            print("[!] Install PyMuPDF first:  py -3.11 -m pip install PyMuPDF")
            sys.exit(1)
        rows = rows_from_folder(args.safe_dir, "safe") + rows_from_folder(args.malicious_dir, "malicious")
        if args.suspicious_dir:
            rows += rows_from_folder(args.suspicious_dir, "suspicious")
        df = pd.DataFrame(rows)
    else:
        print(__doc__)
        print("[!] No dataset supplied. Use --csv, or --safe-dir together with --malicious-dir.")
        sys.exit(1)

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    counts = df["label"].value_counts().to_dict()
    print("Samples per class:", counts)
    if len(counts) < 2 or min(counts.values()) < MIN_PER_CLASS:
        print(f"[!] Need at least 2 classes with >= {MIN_PER_CLASS} samples each.")
        sys.exit(1)

    X, y = df[PDF_FEATURE_NAMES], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(classification_report(y_te, pred, zero_division=0))
    print(confusion_matrix(y_te, pred, labels=list(model.classes_)))

    joblib.dump(model, MODEL_OUT)
    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump({"classes": list(model.classes_), "samples": {k: int(v) for k, v in counts.items()},
                   "report": classification_report(y_te, pred, output_dict=True, zero_division=0)}, fh, indent=2)
    print("Saved", MODEL_OUT)


if __name__ == "__main__":
    main()
