"""
evaluate_models.py
==================
Honest evaluation of the email and PDF machine-learning models.

    python evaluate_models.py            # prints a report, writes model_evaluation.json

Why three kinds of numbers?
  1. RANDOM SPLIT  (what train_*_model.py prints): optimistic. The training data
     are template-generated, so test rows share wording with training rows.
  2. LEAVE-ONE-FAMILY-OUT: for each scam/legit "family" (bank-KYC, CEO fraud,
     delivery scams ...), train on ALL OTHER families and test on the held-out
     one. This measures generalisation to a type of message the model never saw.
  3. INDEPENDENT SAMPLES: hand-written emails / generated fixture PDFs that are
     not produced by make_training_data.py. They are written by the same author,
     so they are still not a substitute for real-world data.

For the independent samples three systems are compared:
    rules only | ML only | rules + ML (what the app uses; the more severe verdict wins)
"""

import json
import os
import tempfile
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import email_pdf_fixtures as F
from email_analyzer import normalize_text, parse_eml_bytes
from email_model import EMAIL_MODEL_PATH, load_email_model, predict_email
from pdf_analyzer import analyze_pdf
from pdf_model import PDF_FEATURE_NAMES, PDF_MODEL_PATH, classify_pdf, load_pdf_model

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "model_evaluation.json")


def email_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True,
                                  stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


# ---------------------------------------------------------------------------
# independent e-mails (different wording/structure than the generator)
# ---------------------------------------------------------------------------
def eml(sender, subject, body, extra=""):
    return f"From: {sender}\nTo: you@example.org\nSubject: {subject}\n{extra}Content-Type: text/plain\n\n{body}\n"


INDEP_LEGIT = F.LEGIT_EMAILS + [
    eml("Dana <dana@bluebird.example>", "Slides for Thursday", "Attached are the slides for Thursday's talk. Tell me if the charts look right to you."),
    eml("Facilities <facilities@harbor-co.example>", "Fire drill", "There will be a fire drill at 11am tomorrow. Please use the stairs and gather at the car park."),
    eml("Pat <pat@kestrel.example>", "Code review", "I left a few comments on your merge request. Nothing blocking, mostly naming. Let's pair on it after lunch."),
    eml("Bank Alerts <alerts@chase.com>", "Your statement is ready", "Your monthly statement is now available. Sign in at https://www.chase.com to view it. We will never ask for your password by email."),
    eml("Neighbourhood Watch <watch@greenfield.example>", "Garden party", "The street garden party is on Saturday afternoon. Bring a dish to share and a chair."),
    eml("Payroll <payroll@northwind.example>", "Payslip available", "Your payslip for this month is available in the HR portal. Contact payroll if the hours look wrong."),
    eml("Support <support@lumina.example>", "Ticket #4821 resolved", "We have resolved your ticket. If the problem returns, reply to this email and we will reopen it."),
    eml("Travel <bookings@orbitech.example>", "Your itinerary", "Your flight departs at 7:40am from terminal 2. Check-in opens three hours before departure."),
]
INDEP_PHISH = F.PHISHING_EMAILS + [
    eml('"Apple ID" <appleid@apple-id-confirm.example>', "Your Apple ID has been disabled", "Your Apple ID was used to sign in from an unknown device. To restore access confirm your password and card number here: http://apple-id-confirm.example/restore"),
    eml("Zoom <meetings@zoom-secure-meet.top>", "You missed a meeting", "You have a missed meeting recording waiting. Log in with your work email password to view: http://zoom-secure-meet.top/rec"),
    eml("Courier <track@parcel-desk.example>", "Delivery attempt failed", "We could not deliver your parcel. Pay the 1.99 redelivery fee and confirm your card number at http://203.0.113.88/redeliver within 24 hours."),
    eml("Director <director.work.office@gmail.com>", "Urgent task", "Are you at your desk? I need you to purchase gift cards for staff rewards and send me the codes. Keep it between us for now."),
    eml("Microsoft 365 <no-reply@m1crosoft-online.example>", "Mailbox storage full", "Your mailbox is full and incoming mail will be rejected. Verify your account and sign in to increase storage: http://m1crosoft-online.example/mail"),
    eml("Tax Office <refund@tax-gov-refund.example>", "Refund approved", "A refund of 640.00 is approved. Submit your bank account number and social security number to receive it. Offer expires in 24 hours."),
    eml("Docs <share@docu-sign-docs.example>", "Please sign the document", "A contract is waiting. Sign in with your email password to review and sign: http://docu-sign-docs.example/sign"),
    eml("Netflix <billing@netflix-membership.example>", "Payment declined", "We could not process your payment. Update your billing information and credit card number to avoid cancellation: http://netflix-membership.example/update"),
]
INDEP_SPAM = [
    eml("Deals <deals@megasale.example>", "WINNER!!! FREE prize", "You are the lucky winner of our lottery! Claim your free gift now, 100% free, no obligation, limited offer!!! Earn money fast, work from home, buy now!"),
]


def eml_results(model):
    """Return classification lists for legit / phishing / spam independent samples."""
    out = {}
    for name, items in (("legit", INDEP_LEGIT), ("phish", INDEP_PHISH), ("spam", INDEP_SPAM)):
        out[name] = [predict_email(parse_eml_bytes(t.encode()), model)["classification"] for t in items]
    return out


def ml_only_labels(model, items):
    labels = []
    for t in items:
        p = parse_eml_bytes(t.encode())
        labels.append(str(model.predict([p["model_text"]])[0]).upper())
    return labels


def summarize_email(res):
    legit_fp = sum(1 for c in res["legit"] if c != "SAFE")
    phish_missed = sum(1 for c in res["phish"] if c in ("SAFE", "SPAM", "INCONCLUSIVE"))
    return {"legit_total": len(res["legit"]), "legit_not_safe(false_alarms)": legit_fp,
            "phish_total": len(res["phish"]), "phish_missed(safe/spam/inconclusive)": phish_missed,
            "phish_flagged_as_PHISHING": sum(1 for c in res["phish"] if c == "PHISHING"),
            "legit_labels": res["legit"], "phish_labels": res["phish"]}


# ---------------------------------------------------------------------------
# PDF independent fixtures (not produced by make_training_data.py)
# ---------------------------------------------------------------------------
def indep_pdfs():
    import make_sample_files as S
    safe = [
        F.text_pdf(["Annual leave request form guidance", "Employees should submit leave requests two weeks in advance.", "Managers approve requests within three working days."]),
        F.text_pdf(["Conference programme", "Opening keynote at 9am followed by parallel sessions on databases and compilers.", "Lunch is served in the main hall."], "flate", links=["https://www.example.com/schedule"]),
        F.text_pdf(["Invoice 7781", "Design services for October. Amount due 450.00 within 14 days.", "Bank account 11223344, IBAN GB00TEST000011223344, SWIFT TESTGB2L."], "a85flate"),
        F.widget_form_pdf(["First name", "Last name", "Department"]),
        S.normal_pdf(),
        F.text_pdf(["Product manual", "Press and hold the power button for three seconds to start the device.", "Charge the battery fully before first use.", "Visit https://docs.python.org/3/ for examples."]),
    ]
    bad = [
        ("malicious", S.malicious_indicators_pdf()),
        ("malicious", F.objstm_javascript_pdf()),
        ("malicious", F.obfuscated_js_pdf()),
        ("malicious", F.text_pdf(["Security notice", "Your bank account is locked. Confirm your identity and enter your OTP and password now.",
                                  "Failure to respond within 24 hours will close the account. Pay the release fee by wire transfer."], "flate")),
        ("malicious", F.text_pdf(["Dropbox shared file", "Sign in with your password to view the document, urgent action required."], links=["http://dr0pbox-login.example/view"])),
        ("suspicious", S.javascript_only_pdf()),
        ("suspicious", S.embedded_file_pdf()),
        ("suspicious", F.text_pdf(F.NORMAL_TEXT, links=["http://203.0.113.9/login"])),
        ("suspicious", F.text_pdf(F.NORMAL_TEXT, links=["http://paypa1.com/verify"])),
        ("suspicious", F.widget_form_pdf(["Name", "Password"])),
        ("suspicious", F.text_pdf(F.NORMAL_TEXT + ["Confirm your account at https://bit.ly/3zzZ"])),
    ]
    return safe, bad


def pdf_labels(model):
    safe, bad = indep_pdfs()

    def run(data):
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.write(fd, data)
        os.close(fd)
        try:
            return analyze_pdf(path)["classification"]
        finally:
            os.remove(path)

    import pdf_analyzer
    old = pdf_analyzer._ML_MODEL
    pdf_analyzer._ML_MODEL = model
    try:
        return [run(d) for d in safe], [(lab, run(d)) for lab, d in bad]
    finally:
        pdf_analyzer._ML_MODEL = old


def main():
    report = {"note": "Synthetic training data. Numbers below are NOT real-world accuracy."}

    # ---------------- EMAIL ----------------
    df = pd.read_csv(os.path.join(BASE, "dataset", "email_dataset.csv"))
    df["x"] = df["text"].astype(str).map(normalize_text)
    fam_res = {}
    for fam in sorted(df["family"].unique()):
        train, test = df[df["family"] != fam], df[df["family"] == fam]
        pipe = email_pipeline().fit(train["x"], train["label"])
        pred = pipe.predict(test["x"])
        true = test["label"].iloc[0]
        if true == "safe":
            fam_res[fam] = {"n": len(test), "metric": "specificity", "value": round(float((pred == "safe").mean()), 4)}
        else:
            fam_res[fam] = {"n": len(test), "metric": "recall(not safe)", "value": round(float((pred != "safe").mean()), 4),
                            "exact_class": round(float((pred == true).mean()), 4)}
    report["email_leave_one_family_out"] = fam_res

    model = load_email_model()
    rules = summarize_email(eml_results(None))
    combined = summarize_email(eml_results(model))
    ml_legit = ml_only_labels(model, INDEP_LEGIT)
    ml_phish = ml_only_labels(model, INDEP_PHISH)
    report["email_independent"] = {
        "rules_only": rules, "rules_plus_ml": combined,
        "ml_only": {"legit_not_safe": sum(1 for c in ml_legit if c != "SAFE"), "legit_total": len(ml_legit),
                    "phish_missed": sum(1 for c in ml_phish if c == "SAFE"), "phish_total": len(ml_phish),
                    "legit_labels": ml_legit, "phish_labels": ml_phish},
    }

    # ---------------- PDF ----------------
    pdf = pd.read_csv(os.path.join(BASE, "dataset", "pdf_features.csv"))
    pfam = {}
    for fam in sorted(pdf["family"].unique()):
        train, test = pdf[pdf["family"] != fam], pdf[pdf["family"] == fam]
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(train[PDF_FEATURE_NAMES], train["label"])
        pred = rf.predict(test[PDF_FEATURE_NAMES])
        true = test["label"].iloc[0]
        pfam[fam] = {"n": len(test), "true": true, "exact_class": round(float((pred == true).mean()), 4),
                     "flagged_not_safe": round(float((pred != "safe").mean()), 4)}
    report["pdf_leave_one_family_out"] = pfam

    pmodel = load_pdf_model()
    s_rules, b_rules = pdf_labels(None)
    s_comb, b_comb = pdf_labels(pmodel)
    report["pdf_independent"] = {
        "rules_only": {"safe_pdfs_labels": s_rules, "bad_pdfs": b_rules},
        "rules_plus_ml": {"safe_pdfs_labels": s_comb, "bad_pdfs": b_comb},
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ---------------- print ----------------
    print("\nEMAIL - leave-one-family-out (train on every other family, test on the unseen one)")
    for fam, r in fam_res.items():
        print(f"  {fam:24s} n={r['n']:4d}  {r['metric']:17s} {r['value']:.3f}" + (f"   exact class {r['exact_class']:.3f}" if "exact_class" in r else ""))
    print("\nEMAIL - independent hand-written samples (%d legit, %d phishing)" % (len(INDEP_LEGIT), len(INDEP_PHISH)))
    for name, r in (("rules only", rules), ("rules + ML", combined)):
        print(f"  {name:11s} legit false alarms: {r['legit_not_safe(false_alarms)']}/{r['legit_total']}   "
              f"phishing missed: {r['phish_missed(safe/spam/inconclusive)']}/{r['phish_total']}   labelled PHISHING: {r['phish_flagged_as_PHISHING']}")
    m = report["email_independent"]["ml_only"]
    print(f"  ML only     legit not-safe: {m['legit_not_safe']}/{m['legit_total']}   phishing called safe: {m['phish_missed']}/{m['phish_total']}")
    print("\nPDF - leave-one-family-out")
    for fam, r in pfam.items():
        print(f"  {fam:22s} n={r['n']:3d} true={r['true']:10s} exact class {r['exact_class']:.3f}  flagged not-safe {r['flagged_not_safe']:.3f}")
    print("\nPDF - independent fixtures")
    for name, key in (("rules only", "rules_only"), ("rules + ML", "rules_plus_ml")):
        r = report["pdf_independent"][key]
        fp = sum(1 for c in r["safe_pdfs_labels"] if c != "SAFE")
        miss = sum(1 for lab, c in r["bad_pdfs"] if c == "SAFE")
        print(f"  {name:11s} ordinary PDFs not SAFE: {fp}/{len(r['safe_pdfs_labels'])}   risky PDFs called SAFE: {miss}/{len(r['bad_pdfs'])}")
        print("              ordinary:", r["safe_pdfs_labels"])
        print("              risky   :", [c for _, c in r["bad_pdfs"]])
    print("\nSaved", OUT)


if __name__ == "__main__":
    main()
