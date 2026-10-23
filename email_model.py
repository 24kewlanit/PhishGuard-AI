"""
email_model.py
==============
STAGE 3 - EMAIL DETECTION (SAFE / SPAM / PHISHING)

There are TWO detection methods in this file. The result always says which
one produced the answer (field: "detection_method"):

  1. REAL MACHINE LEARNING  ->  used only if `email_model.pkl` exists.
     A scikit-learn Pipeline (TF-IDF + LogisticRegression) created by
     `train_email_model.py` from a labeled CSV dataset.

  2. RULE-BASED HEURISTICS  ->  used when `email_model.pkl` does NOT exist.
     This is NOT machine learning. It is a transparent checklist (urgent
     wording, credential wording, suspicious links, sender mismatch ...).
     Every point added to the score has a matching human-readable reason.

The project does not ship an email dataset, so by default method 2 is active.
Once you train a model (see train_email_model.py) the ML pipeline is used
automatically and the rules are kept only as a safety net (see below).

Verdicts (same for both methods)
--------------------------------
    PHISHING     : phishing evidence score 60 - 100
    SUSPICIOUS   : phishing evidence score 25 - 59   (needs human review)
    SPAM         : unwanted/promotional wording only (spam score >= 25)
    SAFE         : the checks that COULD be run found nothing (score < 25)
    INCONCLUSIVE : the email could not be analysed properly (unreadable /
                   malformed input, or no sender information at all) and no
                   strong evidence was found. This is NEVER shown as SAFE.

The risk score is a rule-based evidence score (0-100), NOT a calibrated
probability. A SAFE verdict only means that no indicator was found by the
checks that were actually performed; it is not a guarantee.

Public functions
----------------
    load_email_model(path=None)  -> sklearn Pipeline or None
    predict_email(parsed_email, model=None) -> dict
"""

import os
import re
from urllib.parse import urlparse

import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMAIL_MODEL_PATH = os.path.join(BASE_DIR, "email_model.pkl")

PHISHING_THRESHOLD = 60
SUSPICIOUS_THRESHOLD = 25
SPAM_THRESHOLD = 25
MIN_WORDS_FOR_VERDICT = 3
AUTHENTICATED_MARKETING_MAX_RISK = 40   # display cap for authenticated, non-scam marketing mail

# How the optional ML model is allowed to influence the verdict (see predict_email).
# Chosen from leave-one-family-out measurements (evaluate_models.py): below this
# probability the model produced too many false alarms on message types it had
# never seen. The value is an UNCALIBRATED model score, not a real probability.
ML_MIN_PROB = 0.80
ML_ONLY_RISK = 30            # risk shown when ONLY the model raised the alarm (=> SUSPICIOUS)


# ===========================================================================
# 1) MODEL LOADING
# ===========================================================================
def load_email_model(path=None):
    """
    Load the trained email Pipeline. Returns None if the file does not exist
    or cannot be loaded (the app then uses the rule-based fallback).

    Note: only load .pkl files that YOU created. Pickle files can run code
    when loaded, so never load a model file from an untrusted source.
    """
    path = path or EMAIL_MODEL_PATH
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        # Must be a Pipeline-like object that gives class probabilities.
        if not (hasattr(model, "predict_proba") and hasattr(model, "classes_")):
            print("email_model.pkl ignored: it does not provide predict_proba/classes_.")
            return None
        return model
    except Exception as exc:
        print("email_model.pkl could not be loaded:", repr(exc))
        return None


# ===========================================================================
# 2) RULE-BASED SIGNALS  (used as the detector when no ML model exists)
# ===========================================================================
def _phrase_regex(phrases):
    """One compiled regex that matches any phrase as whole words (longest first)."""
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b")


CREDENTIAL_RE = _phrase_regex([
    "password", "verify your account", "confirm your identity", "login credentials",
    "log in", "sign in", "security code", "one-time password", "otp", "pin",
    "social security", "ssn", "account number", "credit card", "card number",
    "bank details", "update your payment", "billing information",
    "verify your email", "validate your account", "reactivate your account",
    "confirm your account", "update your account", "kyc",
])

URGENCY_RE = _phrase_regex([
    "urgent", "urgently", "immediately", "act now", "within 24 hours",
    "within 48 hours", "final notice", "last warning", "suspended",
    "will be closed", "will be suspended", "unauthorized", "unusual activity",
    "suspicious activity", "verify now", "expires today", "immediate action",
    "action required", "account locked", "has been locked", "security alert",
])

GENERIC_GREETING_RE = _phrase_regex([
    "dear customer", "dear user", "dear member", "dear account holder",
    "dear client", "valued customer", "dear sir/madam", "dear email user",
])

SPAM_RE = _phrase_regex([
    "winner", "you have won", "you won", "congratulations", "lottery", "prize",
    "free gift", "100% free", "free money", "cash bonus", "earn money",
    "make money", "work from home", "get rich", "viagra", "cialis", "pharmacy",
    "cheap", "discount", "limited offer", "limited time", "buy now", "order now",
    "no obligation", "risk free", "casino",
    "double your", "miracle", "weight loss", "credit repair", "loan approved",
    "special promotion", "exclusive deal", "claim your", "free trial",
    "lucky winner", "free spins", "guaranteed returns", "no experience", "cash advance",
    "erase your debt", "no credit check", "once in a lifetime", "100% guaranteed",
    "this is not spam", "dear friend", "act now", "increase your sales",
])

MONEY_RE = re.compile(r"(?i)(\$\s?\d[\d,]{3,}|\b\d[\d,]*\s?(million|billion)\b|\busd\s?\d)")

# ---------------------------------------------------------------------------
# Promotional / marketing detection (shared by the e-mail AND the PDF scanner)
# ---------------------------------------------------------------------------
# Ordinary marketing mail ("Mega Sale - up to 40% off - Shop now") is not a
# scam, but it IS unwanted bulk promotion, so it is reported as SPAM. It is
# recognised by SEVERAL independent signals rather than one keyword, so a
# normal receipt or newsletter that happens to contain one word is not flagged.
PROMO_RE = _phrase_regex([
    "sale", "flash sale", "mega sale", "big sale", "summer sale", "winter sale", "festive sale",
    "festival sale", "end of season sale", "clearance", "black friday", "cyber monday",
    "special offer", "exclusive offer", "best offer", "festive offer", "this offer", "offer ends",
    "offer expires", "best price", "lowest price", "price drop", "price cut", "great deal",
    "hot deal", "best deals", "mega deal", "deal of the day", "lifetime deal", "lifetime plan",
    "lifetime access", "lifetime license", "lifetime offer", "free shipping", "promo code",
    "coupon", "coupon code", "voucher", "cashback", "cash back", "save up to", "save big",
    "save now", "grab yours", "get yours", "don't miss", "don't miss out", "hurry", "ends soon",
    "ends tonight", "last chance", "while stocks last", "while supplies last", "bundle", "upgrade now", "free upgrade", "upgrade today", "subscribe now", "members only",
    "exclusive access", "early bird", "new arrivals", "mega discount", "extra discount",
])
# "21% off", "up to 70%", "save 30%", "50 percent off"
PERCENT_OFF_RE = re.compile(
    r"(?i)(\b\d{1,3}\s?(?:%|percent)\s*(?:off|discount|savings?)\b|\bup\s+to\s+\d{1,3}\s?(?:%|percent)"
    r"|\bsave\s+(?:up\s+to\s+)?\d{1,3}\s?(?:%|percent))")
# prices such as $199, US$ 49.99, EUR 20, Rs. 499, 1,999 INR
PRICE_RE = re.compile(r"(?i)(?:[$\u20ac\u00a3\u20b9]\s?\d[\d,]*(?:\.\d{1,2})?|\b(?:rs\.?|inr|usd|eur)\s?\d[\d,]*(?:\.\d{1,2})?)")
# button / link text that tries to get a click
CTA_RE = _phrase_regex([
    "click here", "learn more", "shop now", "explore more", "view more", "see more", "read more",
    "get started", "sign up now", "view offer", "see offer", "view deal", "claim now", "redeem now",
    "get it now", "start free trial", "try it free", "tap here",
])
# footer lines that only mass mailings have
BULK_FOOTER_RE = re.compile(
    r"(?i)(unsubscribe|you are receiving this (?:e-?mail|message|newsletter)|you received this (?:e-?mail|message)"
    r"|view (?:this (?:e-?mail|message) )?in (?:your )?browser|problems? (?:viewing|reading|seeing) this (?:e-?mail|message)"
    r"|manage (?:your )?(?:e-?mail )?(?:preferences|subscriptions?)|update (?:your )?(?:e-?mail )?preferences"
    r"|opt[- ]out|no longer (?:wish|want) to receive|do not want to receive|don't want to receive"
    r"|(?:please )?do not reply to this (?:e-?mail|letter|message))")

# Brand -> real domains (second-level names). Used to spot impersonation.
BRANDS = {
    "paypal": ["paypal"],
    "microsoft": ["microsoft", "office", "outlook", "live", "microsoftonline", "office365"],
    "apple": ["apple", "icloud"],
    "google": ["google", "gmail", "googlemail", "youtube"],
    "amazon": ["amazon", "amazonaws"],
    "netflix": ["netflix"],
    "facebook": ["facebook", "fb", "meta", "instagram"],
    "instagram": ["instagram", "facebook", "meta"],
    "linkedin": ["linkedin"],
    "dropbox": ["dropbox"],
    "docusign": ["docusign"],
    "dhl": ["dhl"],
    "fedex": ["fedex"],
    "hdfc": ["hdfcbank", "hdfc"],
    "sbi": ["sbi", "onlinesbi"],
    "icici": ["icicibank", "icici"],
    "paytm": ["paytm"],
    "chase": ["chase", "jpmorgan"],
    "wellsfargo": ["wellsfargo"],
    "bankofamerica": ["bankofamerica"],
}

FREE_MAIL = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "proton.me",
    "protonmail.com", "mail.com", "gmx.com", "yandex.com", "rediffmail.com", "live.com",
}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "work", "gq", "tk", "ml", "cf", "ga",
    "icu", "cyou", "rest", "country", "support", "monster", "buzz",
}
DANGEROUS_ATTACHMENT_EXT = {
    "exe", "scr", "js", "vbs", "bat", "cmd", "ps1", "jar", "msi", "lnk", "hta",
    "iso", "img", "dll", "com", "wsf",
}
MACRO_ATTACHMENT_EXT = {"docm", "xlsm", "pptm"}
ARCHIVE_EXT = {"zip", "rar", "7z", "gz"}


def _host(url):
    """Hostname of a URL string (no network access - pure string parsing)."""
    try:
        if not re.match(r"(?i)^https?://", url):
            url = "http://" + url
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_ip(host):
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))


def _sld(host):
    """Second-level name, e.g. 'mail.paypal.com' -> 'paypal', 'amazon.co.uk' -> 'amazon'."""
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return host
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return parts[-3]
    return parts[-2]


def _registered_domain(host):
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _mentions(text, brand):
    """
    Does `text` mention the brand? Long brand names (6+ letters) match anywhere
    ('paypal-secure'); short ones must be a whole word so that 'chase' does not
    match 'purchase' and 'apple' does not match 'pineapple'.
    """
    if len(brand) >= 6:
        return brand in text
    return re.search(rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])", text) is not None


def _brand_mismatch(text, host):
    """If `text` mentions a brand but `host` is not that brand's real domain, return the brand."""
    text = text.lower()
    host_sld = _sld(host)
    for brand, real in BRANDS.items():
        if _mentions(text, brand) and host_sld not in real and host_sld != brand:
            return brand
    return None


def _count_points(count, steps):
    """steps = (points_for_1, points_for_2, points_for_3_or_more)."""
    if count <= 0:
        return 0
    return steps[min(count, 3) - 1]


# ---------------------------------------------------------------------------
# Extra word lists / patterns (added for the accuracy fix)
# ---------------------------------------------------------------------------
# Payment wording is split in two so that ordinary invoices (which list a bank
# account / IBAN) are not treated like fraud:
#   PAYMENT_RISK_RE   - methods/requests typical of scams (gift cards, crypto,
#                       wire transfer, "new bank details", ...)
#   PAYMENT_DETAIL_RE - ordinary payment details (weak evidence on its own)
PAYMENT_RISK_RE = _phrase_regex([
    "wire transfer", "gift card", "gift cards", "bitcoin", "cryptocurrency", "crypto wallet",
    "western union", "moneygram", "upi pin", "new bank details", "updated bank details",
    "change of bank details", "change your bank details", "send the codes",
])
PAYMENT_DETAIL_RE = _phrase_regex([
    "bank account", "bank transfer", "routing number", "iban", "swift code", "upi id",
    "tax refund", "processing fee", "advance fee", "overdue invoice", "payment is overdue",
    "unpaid invoice", "remit payment",
])

SECRECY_RE = _phrase_regex([
    "keep this confidential", "strictly confidential", "do not tell anyone",
    "don't tell anyone", "do not share this", "do not discuss this", "keep this between us",
])

MACRO_LURE_RE = _phrase_regex(["enable macros", "enable content", "enable editing"])

# "open the attached invoice.exe" - a file name with a risky extension in the TEXT
_RISKY_FILE_REF_RE = re.compile(
    r"(?i)\b[\w\-]+(?:\.[\w\-]+)*\.(?:exe|scr|js|vbs|bat|cmd|ps1|jar|msi|lnk|hta|iso|dll|wsf|docm|xlsm|pptm)\b")
_OPEN_CUE_RE = re.compile(r"(?i)\b(attach(?:ed|ment)?|enclosed|open|run|execute|install|download|double[- ]click)\b")
# "submit your bank account number and social security number" - an explicit request for
# sensitive details. A negation shortly before ("never share your password") cancels it.
_ASK_SENSITIVE_RE = re.compile(
    r"(?i)\b(submit|send|provide|enter|confirm|reply with|share|type|supply|give us)\b[^.\n]{0,70}?"
    r"\b(social security|ssn|card number|credit card|account number|password|otp|one-time password|pin|cvv|security code)\b")
_NEGATION_BEFORE_RE = re.compile(r"(?i)\b(never|not|n't|no)\b[^.\n]{0,40}$")
_ARCHIVE_PASSWORD_RE = re.compile(r"(?i)\bpassword\s*(?:is|:)\s*\S+")

_LOGIN_WORDS_IN_HOST = re.compile(
    r"(?i)(login|log-in|signin|sign-in|verify|verification|secure|account|update|confirm|wallet|recover|unlock|billing)")

_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})
_SOFT_WORDS = {
    "secure", "security", "login", "signin", "verify", "verification", "account", "accounts",
    "support", "update", "service", "services", "online", "official", "alert", "alerts", "help",
    "billing", "payment", "payments", "pay", "customer", "care", "center", "portal", "web",
    "mail", "id", "auth", "my", "team", "info",
}


def _levenshtein(a, b):
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lookalike_brand(host):
    """
    If the registered name of `host` is a typo / character-swap look-alike of a
    well-known brand (paypa1, micr0soft, arnazon, gogle ...) but is NOT that
    brand's real domain, return the brand name. Pure string logic.
    """
    if not host or _is_ip(host):
        return None
    sld = _sld(host)
    parts = [p for p in re.split(r"[-_]", sld) if p]
    tokens = {sld, *parts}
    joined = "".join(p for p in parts if p not in _SOFT_WORDS)
    if joined:
        tokens.add(joined)

    nohyphen = sld.replace("-", "")
    for brand, real in BRANDS.items():
        if sld in real or sld == brand:
            continue
        if len(brand) >= 6 and "-" in sld and brand in nohyphen and brand not in sld:
            return brand                                   # e.g. docu-sign-docs  ->  docusign
        for tok in tokens:
            if len(tok) < 4 or tok in real or tok == brand:
                continue
            norm = tok.translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w")
            if len(brand) >= 5 and norm == brand:
                return brand
            if len(brand) >= 6 and len(tok) >= 5 and _levenshtein(norm, brand) == 1:
                return brand
    return None


def extract_domain_from_text(text):
    """If anchor text itself looks like a URL/domain, return its host, else ''."""
    m = re.search(r"(?i)\b((?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,})\b", text or "")
    return _host(m.group(1)) if m else ""


def link_text_mismatch(shown_text, href):
    """Reason string if the visible link text shows a different domain than the real target."""
    shown = extract_domain_from_text(shown_text)
    real = _host(href or "")
    if shown and real and _registered_domain(shown) != _registered_domain(real):
        return f"Misleading link: text shows '{shown}' but it actually points to '{real}'"
    return None


def analyze_urls(urls):
    """
    Check URL strings (never fetched). Returns (points, [reasons]) - a reason is
    only produced for an indicator that was really found. Shared by the email
    and the PDF detectors.
    """
    points, reasons = 0, []

    def add(p, reason):
        nonlocal points
        if reason not in reasons:
            reasons.append(reason)
            points += p

    for url in urls or []:
        host = _host(url)
        if not host:
            continue
        if _is_ip(host):
            add(25, f"Suspicious link: uses a raw IP address instead of a domain ({host})")
        if host in SHORTENERS:
            add(10, f"Suspicious link: URL shortener hides the real destination ({host})")
        if "xn--" in host:
            add(20, f"Suspicious link: punycode (look-alike characters) in domain ({host})")
        if re.match(r"(?i)^https?://[^/]*@", url):
            add(20, "Suspicious link: contains '@' which can disguise the real destination")
        if host.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
            add(10, f"Suspicious link: unusual top-level domain ({host})")
        if not _is_ip(host):
            brand = _brand_mismatch(host, host)
            if brand:
                add(25, f"Suspicious link: domain imitates '{brand}' but is not the real {brand} domain ({host})")
            else:
                look = lookalike_brand(host)
                if look:
                    add(25, f"Look-alike domain: '{host}' resembles '{look}' but is not the real {look} domain")
                elif "-" in _sld(host) and _LOGIN_WORDS_IN_HOST.search(_sld(host)):
                    add(8, f"Suspicious link: domain name contains login/verification wording ({host})")
        if host.count(".") >= 4:
            add(8, f"Suspicious link: unusually many subdomains ({host})")
    return min(points, 60), reasons


def text_phishing_signals(text, urls=(), own_domains=()):
    """
    Wording-based phishing evidence, shared by the email and PDF detectors.
    Returns (signals, flags): signals = [(points, reason), ...];
    flags = {"cred": [...], "urgent": [...], "payment": [...]}.
    Combinations count for more than any single keyword, so one weak keyword
    alone never decides the verdict.
    """
    text = (text or "").lower()
    signals = []

    cred = sorted(set(CREDENTIAL_RE.findall(text)))
    urgent = sorted(set(URGENCY_RE.findall(text)))
    pay_risk = sorted(set(PAYMENT_RISK_RE.findall(text)))
    pay_detail = sorted(set(PAYMENT_DETAIL_RE.findall(text)))
    pay = pay_risk + pay_detail
    flags = {"cred": cred, "urgent": urgent, "payment": pay,
             "payment_risk": pay_risk, "payment_detail": pay_detail}

    if cred:
        signals.append((_count_points(len(cred), (8, 14, 20)),
                        "Credential or sensitive-information wording detected: " + ", ".join(cred[:4])))
    if urgent:
        signals.append((_count_points(len(urgent), (8, 14, 18)),
                        "Urgent or threatening language detected: " + ", ".join(urgent[:4])))
    if pay_risk:
        signals.append((_count_points(len(pay_risk), (14, 22, 28)),
                        "Payment / money-transfer wording typical of scams: " + ", ".join(pay_risk[:4])))
    if pay_detail:
        signals.append((_count_points(len(pay_detail), (4, 8, 10)),
                        "Payment-detail wording: " + ", ".join(pay_detail[:4])))
    if GENERIC_GREETING_RE.search(text):
        signals.append((6, "Generic greeting used instead of your name"))
    if SECRECY_RE.search(text):
        signals.append((8, "Request to keep the message secret (common in impersonation fraud)"))
    if MACRO_LURE_RE.search(text):
        signals.append((15, "Asks you to enable macros/content (a common malware lure)"))

    explicit_ask = None
    for m in _ASK_SENSITIVE_RE.finditer(text):
        if not _NEGATION_BEFORE_RE.search(text[max(0, m.start() - 45):m.start()]):
            explicit_ask = m.group(2)
            break
    if explicit_ask:
        signals.append((15, f"Explicitly asks you to send or enter sensitive details ({explicit_ask})"))
    asks = bool(cred or pay_risk or len(pay_detail) >= 2 or explicit_ask)
    if asks and urgent:
        signals.append((20, "Combination: asks for credentials/payment AND pressures you to act urgently "
                            "(classic phishing pattern)"))

    own = {d for d in own_domains if d}
    external = []
    for u in urls or []:
        host = _host(u)
        if host and _registered_domain(host) not in own:
            external.append(host)
    external = list(dict.fromkeys(external))
    if asks and external:
        signals.append((15, f"Asks for credentials/payment details and links to {len(external)} "
                            f"external domain(s): {', '.join(external[:3])}"))

    # Impersonation: names a well-known brand, wants action, but links go elsewhere
    if (asks or urgent) and external:
        for brand, real in BRANDS.items():
            if not _mentions(text, brand) or brand in {s for s in own}:
                continue
            if any(o in real or o == brand for o in {_sld(d) for d in own}):
                continue
            unrelated = [h for h in external if _sld(h) not in real and _sld(h) != brand]
            if unrelated:
                signals.append((20, f"Possible impersonation of '{brand}': the message refers to it but "
                                    f"links point to unrelated domain(s): {', '.join(unrelated[:2])}"))
                break
    return signals, flags


def _related_domains(a, b):
    """Same registered domain, or the same brand name on another TLD (company.com / company.co.uk)."""
    return _registered_domain(a) == _registered_domain(b) or _sld(a) == _sld(b)


def spam_signals(text, subject="", link_count=0, bulk_headers=()):
    """
    Spam / promotional-mail evidence, shared by the e-mail and PDF detectors.
    Returns [(points, reason), ...]; a reason is only produced for something
    that was really found. Several weak signals together reach the SPAM
    threshold; one lone keyword never does.
    """
    text = (text or "").lower()
    subject = (subject or "").lower()
    out = []

    strong = sorted(set(SPAM_RE.findall(text)))
    if strong:
        out.append((min(35, 9 * len(strong)), "Scam-style or hard-sell wording detected: " + ", ".join(strong[:5])))

    promo = sorted(set(PROMO_RE.findall(text)))
    if promo:
        out.append((min(18, 6 * len(promo)), "Promotional / marketing wording detected: " + ", ".join(promo[:5])))

    pct = PERCENT_OFF_RE.search(text)
    if pct:
        out.append((8, f"Discount / percentage-off offer ('{pct.group(0).strip()}')"))

    prices = {m.group(0).lower().replace(" ", "") for m in PRICE_RE.finditer(text)}
    if len(prices) >= 3:
        out.append((4, f"Price list or plan comparison ({len(prices)} different prices) - typical of sales mail"))

    cta = sorted(set(CTA_RE.findall(text)))
    if cta:
        out.append((min(8, 4 * len(cta)), "Click-bait call-to-action wording: " + ", ".join(cta[:4])))

    if BULK_FOOTER_RE.search(text):
        out.append((6, "Mass-mailing footer found (unsubscribe / 'you are receiving this because' wording)"))

    if subject and (PROMO_RE.search(subject) or PERCENT_OFF_RE.search(subject) or SPAM_RE.search(subject)):
        out.append((5, "Promotional wording in the subject line"))

    bulk = list(bulk_headers or [])
    if bulk:
        out.append((4, "Sent through a bulk-mailing / marketing platform (headers: " + ", ".join(bulk[:3]) + ")"))

    if link_count >= 10:
        out.append((3, f"Link-heavy layout ({link_count} links) - typical of advertising mail"))

    if text.count("!") >= 3:
        out.append((6, "Excessive exclamation marks"))
    words = re.findall(r"[A-Za-z]{3,}", f"{subject} {text}")
    if len(words) >= 10 and sum(1 for w in words if w.isupper()) / len(words) > 0.3:
        out.append((8, "Large share of ALL-CAPS words"))
    if MONEY_RE.search(text):
        out.append((8, "Large money amount mentioned"))

    # Recruitment / internship campaigns are common unsolicited-mail categories.
    # This is a SPAM signal, not proof of fraud: legitimate recruitment messages
    # can receive the same label and should be verified independently.
    recruitment_terms = (
        "internship program", "internship programme", "internship opportunity",
        "internship openings", "internship application", "virtual internship",
        "work from home position", "job offer", "job opportunity",
        "career opportunity", "we are hiring", "congratulations your application",
    )
    recruitment_hits = [phrase for phrase in recruitment_terms if phrase in text]
    if recruitment_hits:
        out.append((16, "Unsolicited recruitment / internship promotion wording detected: "
                        + ", ".join(recruitment_hits[:4])))

    recruitment_pressure = (
        "no interviews", "limited seats", "limited slots", "register now",
        "last date to register", "registration closes", "offer letter",
        "completion certificate", "apply now", "start your career",
    )
    pressure_hits = [phrase for phrase in recruitment_pressure if phrase in text]
    if recruitment_hits and pressure_hits:
        out.append((10, "Recruitment message uses application pressure or benefit claims: "
                        + ", ".join(pressure_hits[:4])))

    return out


def collect_signals(parsed):
    """
    Look for indicators in the parsed email. Returns:
        phishing_signals: [(points, reason), ...]
        spam_signals:     [(points, reason), ...]
    A reason is added ONLY when that indicator was really found.
    """
    phishing, spam = [], []

    subject = parsed.get("subject", "") or ""
    body = parsed.get("body", "") or ""
    text = f"{subject}\n{body}".lower()
    urls = parsed.get("urls", []) or []
    links = parsed.get("links", []) or []
    sender_domain = parsed.get("sender_domain", "")
    sender_name = parsed.get("sender_name", "")
    own = {_registered_domain(sender_domain)} if sender_domain else set()

    # ---------------- wording (+ combinations with links) ----------------
    text_signals, flags = text_phishing_signals(text, urls, own)
    phishing += text_signals
    asks_or_urgent = bool(flags["cred"] or flags["urgent"] or flags["payment"])

    # ---------------- links ----------------
    link_points, link_reasons = analyze_urls(urls)
    # HTML links whose visible text shows one domain but really go to another
    for link in links:
        reason = link_text_mismatch(link.get("text", ""), link.get("href", ""))
        if reason and reason not in link_reasons:
            link_reasons.append(reason)
            link_points = min(60, link_points + 25)
            break
    if link_reasons:
        phishing.append((link_points, None))          # points once ...
        for reason in link_reasons:                   # ... reasons individually
            phishing.append((0, reason))

    # ---------------- sender ----------------
    reply_domain = parsed.get("reply_to_domain", "")
    if reply_domain and sender_domain and not _related_domains(reply_domain, sender_domain):
        phishing.append((25, f"Reply-To domain ({reply_domain}) is unrelated to the sender domain "
                             f"({sender_domain}) - replies would go somewhere else"))

    if sender_domain:
        brand = _brand_mismatch(sender_name, sender_domain) or _brand_mismatch(sender_domain, sender_domain)
        look = None if brand else lookalike_brand(sender_domain)
        if brand:
            phishing.append((25, f"Suspicious sender: claims to be '{brand}' but is sent from {sender_domain}"))
        elif look:
            phishing.append((25, f"Look-alike sender domain: '{sender_domain}' resembles '{look}' "
                                 f"but is not the real {look} domain"))
        elif sender_domain in FREE_MAIL and asks_or_urgent:
            phishing.append((10, f"Account/security/payment wording sent from a free webmail address ({sender_domain})"))
        if sender_domain.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
            phishing.append((10, f"Sender domain uses an unusual top-level domain ({sender_domain})"))
        shown_addr = re.search(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)", sender_name or "")
        if shown_addr and _registered_domain(shown_addr.group(1).lower()) != _registered_domain(sender_domain):
            phishing.append((20, f"Display name shows an address at {shown_addr.group(1).lower()} "
                                 f"but the message is really sent from {sender_domain}"))

    failed = [f"{k}={v}" for k, v in (parsed.get("auth_results") or {}).items()
              if v in {"fail", "softfail", "permerror"}]
    if failed:
        phishing.append((15, "Email authentication failed: " + ", ".join(failed)))

    # ---------------- attachments ----------------
    has_archive = False
    for att in parsed.get("attachments", []) or []:
        name = att.get("filename", "").lower()
        parts = name.split(".")
        ext = parts[-1] if len(parts) > 1 else ""
        if len(parts) > 2 and ext in DANGEROUS_ATTACHMENT_EXT:
            phishing.append((25, f"Attachment '{att['filename']}' uses a double extension to hide an executable type"))
        if ext in DANGEROUS_ATTACHMENT_EXT:
            phishing.append((35, f"Attachment '{att['filename']}' is an executable/script file type (.{ext})"))
        elif ext in MACRO_ATTACHMENT_EXT:
            phishing.append((20, f"Attachment '{att['filename']}' is a macro-enabled Office file (.{ext})"))
        elif ext in {"html", "htm"}:
            phishing.append((15, f"Attachment '{att['filename']}' is an HTML file (often used to steal logins)"))
        elif ext in ARCHIVE_EXT:
            has_archive = True
            phishing.append((8, f"Attachment '{att['filename']}' is a compressed archive (.{ext})"))
    if has_archive and _ARCHIVE_PASSWORD_RE.search(body):
        phishing.append((10, "Archive attachment with its password given in the message (used to evade scanners)"))

    refs = sorted(set(m.group(0).lower() for m in _RISKY_FILE_REF_RE.finditer(text)))
    if refs and _OPEN_CUE_RE.search(text):
        phishing.append((30, "The message tells you to open/run a file with an executable, script or macro "
                             "extension: " + ", ".join(refs[:3])))

    # ---------------- spam / promotional ----------------
    # Count the distinct link targets: visible URLs + hidden HTML href targets.
    link_count = len({*(urls or []), *(l.get("href", "") for l in links)})
    spam += spam_signals(text, subject=subject, link_count=link_count,
                         bulk_headers=parsed.get("bulk_headers") or [])

    return phishing, spam


def _sum(signals):
    return min(100, sum(p for p, _ in signals))


def _reasons(signals):
    return [r for _, r in signals if r]


# ---------------------------------------------------------------------------
# Coverage: which checks were really possible? (missing info is never "safe")
# ---------------------------------------------------------------------------
def assess_coverage(parsed):
    """
    Returns (checks_performed, blocking_limits, notes).

    blocking_limits are reasons the email CANNOT be called safe even if no
    indicator was found (unreadable input, no sender information, almost no
    content). notes are limitations that do not by themselves block a verdict
    (for example, missing SPF/DKIM/DMARC headers).
    """
    checks, blocking, notes = [], [], []
    subject = parsed.get("subject", "") or ""
    body = parsed.get("body", "") or ""
    urls = parsed.get("urls", []) or []
    atts = parsed.get("attachments", []) or []

    if not parsed.get("parse_ok", True):
        blocking += list(parsed.get("parse_issues") or ["The email could not be parsed."])
        return checks, blocking, notes

    words = len(re.findall(r"\w+", f"{subject} {body}"))
    if words < MIN_WORDS_FOR_VERDICT and not urls and not atts:
        blocking.append("The email contains almost no readable content to analyse.")
    else:
        checks.append("Message wording (credential, payment and urgency requests)")
        checks.append(f"Links ({len(urls)} found): look-alike, shortener, IP-address and mismatch checks")
        checks.append(f"Attachments ({len(atts)} listed): file types and names")

    if parsed.get("has_sender"):
        checks.append("Sender address and domain (impersonation, look-alike domain)")
        if parsed.get("reply_to"):
            checks.append("Reply-To compared with sender")
        else:
            notes.append("No Reply-To header, so there was nothing to compare with the sender.")
    else:
        blocking.append("No sender address was available, so the sender, Reply-To and "
                        "authentication checks could not be performed. Paste the full email "
                        "with headers or upload the .eml file for a complete check.")

    auth = parsed.get("auth_results") or {}
    if auth:
        checks.append("Authentication results reported by the mail server: "
                      + ", ".join(f"{k}={v}" for k, v in auth.items()))
    elif parsed.get("has_headers"):
        notes.append("SPF/DKIM/DMARC results are not present in the headers, so they were not evaluated "
                     "(this is neither a good nor a bad sign by itself).")

    for issue in parsed.get("parse_issues") or []:
        if issue.startswith("The message structure is malformed"):
            notes.append(issue + " Some content may not have been read.")
    return checks, blocking, notes


SEVERITY = {"SAFE": 0, "INCONCLUSIVE": 1, "SPAM": 2, "SUSPICIOUS": 3, "PHISHING": 4}


def heuristic_predict(parsed):
    """Rule-based classification. Returns classification, risk and reasons."""
    phishing, spam = collect_signals(parsed)
    phish_score, spam_score = _sum(phishing), _sum(spam)
    checks, blocking, notes = assess_coverage(parsed)
    evidence = _reasons(phishing) + _reasons(spam)
    extra_note = None

    if phish_score >= PHISHING_THRESHOLD:
        label, risk = "PHISHING", phish_score
    elif phish_score >= SUSPICIOUS_THRESHOLD:
        label, risk = "SUSPICIOUS", phish_score
    elif spam_score >= SPAM_THRESHOLD:
        label, risk = "SPAM", min(max(phish_score, spam_score), PHISHING_THRESHOLD - 1)
        # Genuine marketing mail (sender passed SPF/DKIM/DMARC, no scam wording, no phishing
        # evidence) is still SPAM (unwanted bulk promotion) but is not a scam, so its score is
        # kept moderate instead of looking as dangerous as a lottery fraud.
        auth = parsed.get("auth_results") or {}
        authenticated = bool(auth) and "pass" in auth.values() and all(v == "pass" for v in auth.values())
        scam_style = any(r.startswith("Scam-style") for r in _reasons(spam))
        if authenticated and not scam_style and phish_score == 0:
            risk = min(risk, AUTHENTICATED_MARKETING_MAX_RISK)
            extra_note = ("The sender passed email authentication (" +
                          ", ".join(f"{k}={v}" for k, v in auth.items()) +
                          ") and no scam or phishing indicator was found, so this looks like genuine bulk "
                          "marketing mail rather than fraud. It is labelled SPAM because it is promotional "
                          "mass mailing; the score is kept moderate for that reason.")
    elif blocking:
        label, risk = "INCONCLUSIVE", None
    else:
        label, risk = "SAFE", max(phish_score, spam_score)

    reasons = list(evidence)
    if label == "INCONCLUSIVE":
        reasons = ["Inconclusive: " + b for b in blocking] + evidence
        if not evidence:
            reasons.append("No suspicious indicator was found in the part of the email that could be "
                           "analysed, but that is not enough to call it safe.")
    else:
        notes = blocking + notes          # limitations are still shown, as notes
    if extra_note:
        notes = [extra_note] + notes

    status = "failed" if not parsed.get("parse_ok", True) else ("partial" if blocking else "complete")
    return {
        "classification": label,
        "risk_score": None if risk is None else int(risk),
        "evidence_score": int(max(phish_score, spam_score)),
        "reasons": reasons,
        "notes": notes,
        "checks_performed": checks,
        "analysis_status": status,
        "phishing_signal_score": phish_score,
        "spam_signal_score": spam_score,
    }


# ===========================================================================
# 3) PUBLIC ENTRY POINT
# ===========================================================================
def predict_email(parsed, model=None):
    """
    Classify a parsed email (from email_analyzer) as
    SAFE / SPAM / SUSPICIOUS / PHISHING / INCONCLUSIVE.

    Without a trained model the rule-based heuristics decide. If `model` (a
    trained Pipeline) is given it acts as a second opinion that can only raise
    concern (and only when confident, see ML_MIN_PROB), so it can never hide a
    rule finding and incomplete analysis can never become SAFE.

    The risk score is a rule-based evidence score, not a calibrated probability.
    """
    rules = heuristic_predict(parsed)
    base = {
        "classification": rules["classification"],
        "risk_score": rules["risk_score"],
        "evidence_score": rules["evidence_score"],
        "reasons": rules["reasons"],
        "notes": rules["notes"],
        "checks_performed": rules["checks_performed"],
        "analysis_status": rules["analysis_status"],
    }

    if model is None or not parsed.get("parse_ok", True):
        base.update({
            "detection_method": "Rule-based heuristics (no trained email model installed - this is NOT machine learning)",
            "ml_used": False,
            "probabilities": None,
        })
        return base

    # ------------------------- REAL ML PATH -------------------------
    # The model is a SECOND OPINION that can only RAISE concern - it can never
    # lower a rule finding or turn an incomplete analysis into SAFE, and on its
    # own it can raise a message to SUSPICIOUS (human review), never to PHISHING.
    classes = [str(c).lower() for c in model.classes_]
    proba = model.predict_proba([parsed["model_text"]])[0]
    probs = dict(zip(classes, (float(p) for p in proba)))
    p_phish, p_spam = probs.get("phishing", 0.0), probs.get("spam", 0.0)

    label, risk = rules["classification"], rules["risk_score"]
    reasons = list(rules["reasons"])
    shown = f"uncalibrated model score, trained on synthetic data: phishing {p_phish * 100:.0f}%, spam {p_spam * 100:.0f}%"

    if p_phish >= ML_MIN_PROB:
        if label == "SUSPICIOUS":
            label, risk = "PHISHING", max(risk or 0, PHISHING_THRESHOLD)
            reasons.insert(0, f"Machine-learning model agrees with the rule findings and rates it phishing ({shown})")
        elif label in ("SAFE", "INCONCLUSIVE", "SPAM"):
            label, risk = "SUSPICIOUS", max(risk or 0, ML_ONLY_RISK)
            reasons.insert(0, f"Only the machine-learning model raised an alarm (no rule found a concrete indicator): "
                              f"it rates this like phishing ({shown}). Treat as 'needs review', not proof.")
        else:
            reasons.insert(0, f"Machine-learning model also rates it phishing ({shown})")
    elif p_spam >= ML_MIN_PROB and label in ("SAFE", "INCONCLUSIVE"):
        label, risk = "SPAM", max(risk or 0, SPAM_THRESHOLD)
        reasons.insert(0, f"Machine-learning model rates this as spam ({shown})")
    elif label in ("SAFE", "INCONCLUSIVE") and (p_phish >= 0.5 or p_spam >= 0.5):
        reasons.insert(0, f"Note: the machine-learning model leans towards a problem but is not confident enough "
                          f"to change the result ({shown})")
    elif label in ("SUSPICIOUS", "PHISHING") and p_phish < 0.2 and p_spam < 0.2:
        reasons.append(f"Note: the machine-learning model did not flag this message ({shown}); the rule "
                       f"findings above still stand")

    base.update({
        "classification": label,
        "risk_score": None if label == "INCONCLUSIVE" else int(round(min(100, risk if risk is not None else 0))),
        "reasons": reasons,
        "detection_method": "Rules + machine-learning second opinion (TF-IDF + logistic regression trained on synthetic "
                            "data; the model can only raise concern, never lower it)",
        "ml_used": True,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    })
    return base
