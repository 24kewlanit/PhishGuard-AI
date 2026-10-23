"""
pdf_model.py
============
STAGE 5 - PDF DETECTION (SAFE / SUSPICIOUS / POTENTIALLY MALICIOUS)

This module takes the FACTS found by pdf_analyzer.py (is there JavaScript?
an OpenAction? embedded files? ...) and turns them into a classification.

TWO METHODS (the result says which one was used: "detection_method"):

  1. STATIC RULES  (default - this is NOT machine learning)
     A transparent point system. Each indicator that was really found adds
     points and a human-readable reason.

  2. MACHINE LEARNING (only if `pdf_model.pkl` exists)
     A scikit-learn model trained by `train_pdf_model.py` on features
     extracted from labeled PDFs. No PDF dataset ships with this project, so
     by default only method 1 is active. When a model exists, the final risk
     is the HIGHER of the ML estimate and the static-rule score, so the model
     can never hide a hard indicator like a Launch action.

This file never uses `phishing_model.pkl` - that model is for URLs only.

Score bands
-----------
    SAFE                  :  0 - 24   (only when the analysis was COMPLETE)
    SPAM                  : spam score >= 25 and NO dangerous content (lottery / casino /
                            get-rich / sales-flyer wording). Never used when the PDF is
                            SUSPICIOUS or POTENTIALLY MALICIOUS - those take priority.
    SUSPICIOUS            : 25 - 59
    POTENTIALLY MALICIOUS : 60 - 100
    INCONCLUSIVE          : the PDF could not be inspected adequately (corrupt,
                            password-protected, image-only/scanned with no OCR,
                            no extractable text, parser failure ...) and no
                            other evidence was found. NEVER shown as SAFE.

The score is a rule-based evidence score, not a calibrated probability.

Why JavaScript alone is only "SUSPICIOUS": many harmless PDF forms contain
JavaScript for calculations. It becomes "POTENTIALLY MALICIOUS" mainly when it
is combined with auto-run behaviour (OpenAction / AA) or Launch actions.
"""

import os
import re

import joblib

from email_model import (CREDENTIAL_RE, PAYMENT_DETAIL_RE, PAYMENT_RISK_RE, SHORTENERS, SUSPICIOUS_TLDS,
                         URGENCY_RE, _host, _is_ip, analyze_urls, lookalike_brand, spam_signals,
                         text_phishing_signals)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_MODEL_PATH = os.path.join(BASE_DIR, "pdf_model.pkl")

SUSPICIOUS_THRESHOLD = 25
MALICIOUS_THRESHOLD = 60
SPAM_THRESHOLD = 25          # promotional / spam-only PDFs (no dangerous content)

# The optional ML model is a weak second opinion (its labels come from the same kind of signals the
# rules use and it generalises poorly to unseen PDF types - see evaluate_models.py). It may only
# RAISE a SAFE result to SUSPICIOUS, and only when very confident. It can never lower anything.
PDF_ML_MIN_PROB = 0.95

# Feature columns used for the optional ML model (shared with train_pdf_model.py)
PDF_FEATURE_NAMES = [
    "page_count", "file_size_kb", "encrypted", "javascript", "javascript_count",
    "embedded_files", "dangerous_embedded_files", "url_count", "suspicious_object_count",
    "open_action", "additional_actions", "launch_action", "submit_form",
    "import_data", "rich_media", "xfa", "goto_remote", "obfuscated_names",
    "header_offset", "parse_failed",
    # text / URL / form features (added so the model can learn phishing PDFs, not only active content)
    "text_chars_k", "cred_count", "urgent_count", "payment_risk_count", "payment_detail_count",
    "lookalike_url_count", "ip_url_count", "shortener_url_count", "bad_tld_url_count",
    "link_mismatch_count", "sensitive_form_fields", "form_field_count", "hidden_text", "image_only",
]

# File types that should never be hidden inside a PDF
DANGEROUS_EMBEDDED_EXT = {
    "exe", "scr", "js", "vbs", "bat", "cmd", "ps1", "jar", "msi", "dll", "lnk",
    "hta", "docm", "xlsm", "pptm", "iso", "com", "wsf",
}
URL_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly"}


_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)(pass(word|wd)?|pwd|\bpin\b|otp|cvv|cvc|card|ssn|aadhaar|secret|security.?code|account.?(no|num))")


def findings_to_features(f):
    """Convert analyzer findings into the numeric feature dict used by the ML model."""
    text = (f.get("text") or "").lower()
    hosts = [_host(u) for u in f.get("urls", [])]
    hosts = [h for h in hosts if h]
    fields = [n for n in f.get("form_fields", []) if n]
    return {
        "text_chars_k": round(min(f.get("text_chars", 0), 50_000) / 1000, 2),
        "cred_count": len(set(CREDENTIAL_RE.findall(text))),
        "urgent_count": len(set(URGENCY_RE.findall(text))),
        "payment_risk_count": len(set(PAYMENT_RISK_RE.findall(text))),
        "payment_detail_count": len(set(PAYMENT_DETAIL_RE.findall(text))),
        "lookalike_url_count": sum(1 for h in hosts if lookalike_brand(h)),
        "ip_url_count": sum(1 for h in hosts if _is_ip(h)),
        "shortener_url_count": sum(1 for h in hosts if h in SHORTENERS),
        "bad_tld_url_count": sum(1 for h in hosts if h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS),
        "link_mismatch_count": len(f.get("link_mismatches", [])),
        "sensitive_form_fields": sum(1 for n in fields if _SENSITIVE_FIELD_RE.search(n)),
        "form_field_count": len(fields),
        "hidden_text": int(f.get("hidden_chars", 0) >= 30),
        "image_only": int(bool(f.get("image_only"))),
        **_base_features(f),
    }


def _base_features(f):
    return {
        "page_count": f.get("pages", 0),
        "file_size_kb": round(f.get("file_size", 0) / 1024, 2),
        "encrypted": int(bool(f.get("encrypted"))),
        "javascript": int(bool(f.get("javascript"))),
        "javascript_count": f.get("javascript_count", 0),
        "embedded_files": f.get("embedded_files", 0),
        "dangerous_embedded_files": len(f.get("dangerous_embedded_names", [])),
        "url_count": len(f.get("urls", [])),
        "suspicious_object_count": sum(f.get("object_counts", {}).values()),
        "open_action": int(f.get("open_action", 0) > 0),
        "additional_actions": int(f.get("additional_actions", 0) > 0),
        "launch_action": int(f.get("launch_action", 0) > 0),
        "submit_form": int(f.get("submit_form", 0) > 0),
        "import_data": int(f.get("import_data", 0) > 0),
        "rich_media": int(f.get("rich_media", 0) > 0),
        "xfa": int(f.get("xfa", 0) > 0),
        "goto_remote": int(f.get("goto_remote", 0) > 0),
        "obfuscated_names": len(f.get("obfuscated_names", [])),
        "header_offset": f.get("header_offset", 0),
        "parse_failed": int(not f.get("parse_ok", True)),
    }


def load_pdf_model(path=None):
    """Load pdf_model.pkl if it exists, otherwise return None (rules only)."""
    path = path or PDF_MODEL_PATH
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            return model
    except Exception as exc:
        print("pdf_model.pkl could not be loaded:", repr(exc))
    return None


_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)(pass(word|wd)?|pwd|\bpin\b|otp|cvv|cvc|card|ssn|aadhaar|secret|security.?code|account.?(no|num))")


def rule_based_score(f):
    """
    The static rule engine.
    Returns (risk_score 0-100, [reasons]). A reason is only added for an
    indicator that was actually found in the file.
    """
    score, reasons = 0, []

    def add(points, reason):
        nonlocal score
        score += points
        reasons.append(reason)

    js = bool(f.get("javascript"))
    auto = f.get("open_action", 0) > 0 or f.get("additional_actions", 0) > 0

    # ---------------- active content ----------------
    if js:
        add(35, f"JavaScript detected in the PDF ({f.get('javascript_count', 0)} reference(s))")
    if f.get("launch_action", 0) > 0:
        add(40, "Launch action detected (can start an external program or file)")
    if f.get("open_action", 0) > 0:
        add(10, "OpenAction detected (action runs automatically when the PDF is opened)")
    if f.get("additional_actions", 0) > 0:
        add(10, "Additional Actions (/AA) detected (actions triggered by events)")
    if auto and (js or f.get("launch_action", 0) > 0):
        add(20, "Auto-run action combined with JavaScript/Launch (classic malicious-PDF pattern)")
    if f.get("submit_form", 0) > 0:
        add(15, "SubmitForm action detected (can send form data to a server)")
    if f.get("import_data", 0) > 0:
        add(15, "ImportData action detected")
    if f.get("rich_media", 0) > 0:
        add(20, "RichMedia content detected (embedded Flash/multimedia)")
    if f.get("xfa", 0) > 0:
        add(10, "XFA form detected (dynamic XML forms, historically abused)")
    if f.get("goto_remote", 0) > 0:
        add(10, "Remote GoTo action detected (opens another file/location)")

    # ---------------- embedded files / attachments ----------------
    if f.get("embedded_files", 0) > 0:
        add(30, f"Embedded file detected ({f['embedded_files']} file(s))")
    dangerous = f.get("dangerous_embedded_names", [])
    if dangerous:
        add(25, "Embedded file with a dangerous type: " + ", ".join(dangerous[:3]))
    if f.get("file_attachment", 0) > 0 and not f.get("embedded_files", 0):
        add(20, "File-attachment annotation detected (a file can be hidden behind an icon)")

    # ---------------- links ----------------
    urls = f.get("urls", [])
    if urls:
        add(5, f"PDF contains {len(urls)} external link(s)/URL(s)")
        if len(urls) > 10:
            add(5, "PDF contains an unusually large number of links")
        points, url_reasons = analyze_urls(urls)
        if url_reasons:
            add(points, url_reasons[0])
            reasons.extend(url_reasons[1:])
    for mismatch in f.get("link_mismatches", [])[:3]:
        add(25 if mismatch == f["link_mismatches"][0] else 0, mismatch)

    # ---------------- wording in the visible text ----------------
    text = f.get("text", "") or ""
    if text.strip():
        text_signals, _flags = text_phishing_signals(text, urls)
        for points, reason in text_signals:
            if reason:
                add(points, reason)

    # ---------------- forms / interactive elements ----------------
    fields = [n for n in f.get("form_fields", []) if n]
    sensitive = sorted({n for n in fields if _SENSITIVE_FIELD_RE.search(n)})
    if sensitive:
        add(30, "Form asks for sensitive information (field names: " + ", ".join(sensitive[:4]) + ")")
    elif fields or f.get("acroform", 0) > 0:
        add(5, f"Interactive form fields present ({len(fields)}) - review before filling anything in")
    if f.get("hidden_chars", 0) >= 30:
        add(10, f"Hidden text detected ({f['hidden_chars']} characters in white or tiny type)")

    # ---------------- structure ----------------
    if f.get("obfuscated_names"):
        add(15, "Obfuscated PDF keywords detected (hex-encoded names hiding: "
                + ", ".join(f["obfuscated_names"][:3]) + ")")
    if f.get("header_offset", 0) > 0:
        add(10, "Junk data before the PDF header (a known scanner-evasion trick)")
    if f.get("repaired"):
        add(10, "The PDF was damaged and had to be repaired by the parser (possibly deliberate)")
    if f.get("encryption_blocks_analysis"):
        reasons.append("PDF is encrypted/password-protected: its contents could not be inspected "
                       "(attackers use this to evade scanners, but many legitimate PDFs are protected too)")
    elif f.get("encrypted"):
        reasons.append("PDF uses encryption but could be opened and inspected")
    if not f.get("parse_ok", True) and not f.get("encryption_blocks_analysis"):
        add(10, "PDF structure is damaged or malformed and could not be fully parsed")
    if f.get("jbig2", 0) > 0:
        add(5, "JBIG2-compressed image present (a codec with a history of vulnerabilities)")

    return min(100, score), reasons


def pdf_spam_score(f):
    """
    Spam / promotional evidence in a PDF (lottery, casino, get-rich, sales flyer ...).
    Uses the SAME shared detector as the e-mail scanner, so both agree on what spam is.
    Returns (score 0-100, [reasons]).
    """
    text = f.get("text", "") or ""
    if not text.strip():
        return 0, []
    signals = spam_signals(text, link_count=len(f.get("urls", []) or []))
    return min(100, sum(p for p, _ in signals)), [r for _, r in signals if r]


def _label(risk):
    if risk >= MALICIOUS_THRESHOLD:
        return "POTENTIALLY MALICIOUS"
    if risk >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "SAFE"


def classify_pdf(findings, model=None):
    """
    Classify a PDF from analyzer findings.
    Returns {"classification", "risk_score", "reasons", "detection_method", ...}

    SAFE is only possible when no indicator was found AND the analysis was
    complete. If something prevented a complete analysis (see
    findings["analysis_issues"]) and nothing else was found, the result is
    INCONCLUSIVE with risk_score None.
    """
    risk, reasons = rule_based_score(findings)
    method = "Static rule-based analysis (no trained PDF model installed - this is NOT machine learning)"
    ml_probs = None

    if model is not None:
        import pandas as pd
        feats = findings_to_features(findings)
        X = pd.DataFrame([[feats[c] for c in PDF_FEATURE_NAMES]], columns=PDF_FEATURE_NAMES)
        probs = dict(zip([str(c).lower() for c in model.classes_], model.predict_proba(X)[0]))
        ml_probs = {k: round(float(v), 4) for k, v in probs.items()}
        p_bad = probs.get("malicious", 0) + probs.get("suspicious", 0)
        shown = (f"uncalibrated model score, trained on synthetic data: {probs.get('malicious', 0) * 100:.0f}% "
                 f"malicious / {probs.get('suspicious', 0) * 100:.0f}% suspicious")
        if p_bad >= PDF_ML_MIN_PROB and risk < SUSPICIOUS_THRESHOLD:
            risk = SUSPICIOUS_THRESHOLD
            reasons.insert(0, f"Only the machine-learning model raised an alarm (no rule found a concrete "
                              f"indicator) ({shown}). Treat as 'needs review', not proof.")
        elif probs.get("malicious", 0) >= PDF_ML_MIN_PROB and SUSPICIOUS_THRESHOLD <= risk < MALICIOUS_THRESHOLD:
            risk = MALICIOUS_THRESHOLD
            reasons.insert(0, f"Machine-learning model agrees with the rule findings and rates it malicious ({shown})")
        method = ("Static rules + machine-learning second opinion (random forest trained on synthetic PDFs; "
                  "it can only raise concern, never lower it)")

    risk = int(min(100, risk))
    label = _label(risk)
    issues = list(findings.get("analysis_issues", []))
    notes = []

    # Spam / promotional PDF: nothing dangerous in it, but the text is lottery / casino /
    # get-rich / sales-flyer wording. (Dangerous content always takes priority over SPAM.)
    spam_pts, spam_reasons = pdf_spam_score(findings)
    if label == "SAFE" and spam_pts >= SPAM_THRESHOLD:
        label = "SPAM"
        risk = int(min(max(risk, spam_pts), MALICIOUS_THRESHOLD - 1))
        reasons = spam_reasons + reasons

    if label == "SAFE" and issues:
        label = "INCONCLUSIVE"
        reasons = ["Inconclusive: " + i for i in issues] + reasons
        if not any(not r.startswith("Inconclusive") for r in reasons):
            reasons.append("No malicious indicator was found in the part that could be inspected, "
                           "but that is not enough to call the PDF safe.")
    else:
        notes = issues            # limitations are still shown when evidence decided the verdict

    return {
        "classification": label,
        "risk_score": None if label == "INCONCLUSIVE" else risk,
        "evidence_score": risk,
        "spam_score": int(spam_pts),
        "reasons": reasons,
        "notes": notes,
        "detection_method": method,
        "ml_used": model is not None,
        "probabilities": ml_probs,
    }
