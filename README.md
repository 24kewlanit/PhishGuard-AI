# PhishGuard AI — Hybrid Website Phishing Detector

This version combines **machine learning + live browser behaviour analysis**.

## What was fixed

- Correct PhiUSIIL label mapping: `0 = Phishing`, `1 = Legitimate`.
- Removed the dangerous trusted-domain override that could hide a high-confidence phishing result.
- Added brand-impersonation detection (`google-login-...`, `secure-paypal-...`, etc.).
- Added punycode detection.
- Added live browser signals: redirects, cross-domain forms, credential fields, iframes, network domains, downloads and popups.
- Browser runs headlessly so Flask does not open a second visible browser.
- Cross-domain comparisons use registered-domain logic instead of comparing raw `netloc` strings.
- Added a model sanity-test script.

## Install

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Test the model

```bash
python test_detector.py
```

## Important

No phishing detector can correctly classify **every possible website** from a URL alone. Real-world detection is affected by newly registered domains, compromised legitimate sites, cloaking, bot detection, login walls, regional content and websites that change after loading.

PhishGuard therefore uses several independent signals instead of relying on one rule. It is designed as a **student-project sandbox-style detector**, not as a replacement for Google Safe Browsing, Microsoft Defender, VirusTotal or a commercial threat-intelligence platform.

The browser layer does not enter credentials, submit forms or intentionally interact with page controls.
