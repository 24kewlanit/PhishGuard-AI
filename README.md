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


---

# NEW FEATURE: Email + PDF Security Scanner

Scans an email (`.eml` file or pasted text) together with a PDF attachment and
produces one overall verdict: **SAFE / SPAM / PHISHING / SUSPICIOUS**.
The URL scanner above is unchanged and independent.

## How it works

| Part | File | Method |
|------|------|--------|
| Email parsing | `email_analyzer.py` | Python `email` library (nothing executed, URLs never visited) |
| Email detection | `email_model.py` | **Real ML** (TF-IDF + scikit-learn) if `email_model.pkl` exists, otherwise **rule-based heuristics** (clearly labelled, not ML) |
| PDF analysis | `pdf_analyzer.py` | PyMuPDF static parsing + raw keyword scan (the PDF is parsed, never executed) |
| PDF detection | `pdf_model.py` | **Static rules** by default; ML only if `pdf_model.pkl` exists |
| Combining | `combined_analyzer.py` | 60% email + 40% PDF, plus override rules |
| API | `email_pdf_routes.py` | `POST /email-pdf-scan` |

`phishing_model.pkl` is the **URL** model and is never used for emails or PDFs.

## Datasets

No labeled email or PDF dataset is included. Until you train models, the email
and PDF detectors use rule-based/static analysis (the UI says so).

* Email ML: put a CSV with columns `text,label` (labels `safe`, `spam`, `phishing`)
  at `dataset/email_dataset.csv`, then run `py -3.11 train_email_model.py`.
  All three labels are required (spam/ham-only data cannot detect phishing).
* PDF ML (optional): `py -3.11 train_pdf_model.py --safe-dir ... --malicious-dir ...`

## Install / run / test (Windows)

```bat
py -3.11 -m pip install -r requirements.txt
py -3.11 -m playwright install chromium
py -3.11 app.py
py -3.11 test_email_pdf.py
py -3.11 make_sample_files.py
```

---

## Email + PDF scanner: accuracy update

**Result labels.** Email: `SAFE / SPAM / SUSPICIOUS / PHISHING / INCONCLUSIVE`. PDF: `SAFE / SPAM / SUSPICIOUS / POTENTIALLY MALICIOUS / INCONCLUSIVE`.

* `SPAM` (email **and** PDF) means unwanted promotional / bulk content: lottery and casino wording, get-rich offers, and ordinary marketing mail (sales, "% off", price lists, "Learn more" buttons, unsubscribe footers, marketing-platform headers). It is detected from several signals together, so one stray word (e.g. an invoice that lists prices) does not trigger it. Genuine marketing mail from an authenticated sender (SPF/DKIM/DMARC pass, no scam wording) is still labelled SPAM but keeps a moderate score and an explanatory note. Dangerous content (phishing wording, malicious PDF objects) always takes priority over SPAM.

* `INCONCLUSIVE` means the content could not be analysed adequately (unreadable or malformed input, no sender information, corrupt / password-protected / scanned image-only PDF, no extractable text, parser failure ...). It is **never** shown as Safe and has no risk score (`N/A`).
* `SAFE` only means that no indicator was found by the checks that were actually performed (the card lists them). It is not a guarantee.
* Scores are rule-based evidence points, **not calibrated probabilities**.
* PDFs: the visible text and every URL (link annotations, URI actions, URLs typed in the text) are now checked for phishing wording, fake payment requests, look-alike domains and impersonation. A built-in, dependency-free scanner is always used; PyMuPDF (in `requirements.txt`) adds page-accurate text, link-text vs destination, form fields, annotations and hidden-text checks. OCR is not available, so image-only PDFs are reported as INCONCLUSIVE.

**Tests** (no server needed, nothing is executed or visited):

```bash
python -m unittest test_email_pdf_accuracy -v   # email + PDF + API + frontend rendering (needs Node.js for the last group)
python test_email_pdf.py                        # original end-to-end checks
```

### Machine learning for Email + PDF (trained on SYNTHETIC data)

`email_model.pkl` (TF-IDF + logistic regression) and `pdf_model.pkl` (random forest on static PDF features, incl. text/URL/form features) are included and loaded automatically.

**Be honest about what they are.** No real labeled email/PDF corpus was available when they were trained, so `make_training_data.py` generated one from many templates ("families"). Accuracy on a random split of that data (100% for email) is **not** real-world accuracy. `evaluate_models.py` (results in `model_evaluation.json`) also measures leave-one-family-out generalisation: the email model is strong on some unseen scam types (bank-KYC, credentials, gov-refund, malware > 99%) but weak on others (CEO fraud ~9%, shared-doc ~66%) and gives false alarms on unseen legitimate mail types. The PDF model generalises poorly to unseen PDF types.

Therefore the models are only a **second opinion** on top of the rules:
* they can only **raise** concern - never lower a rule finding, never turn INCONCLUSIVE into SAFE;
* by themselves they can only reach **SUSPICIOUS** ("needs review"), never PHISHING / MALICIOUS;
* they act only when confident (email >= 0.80, PDF >= 0.95; chosen from the held-out measurements);
* model scores shown in the UI are uncalibrated and say so.

To improve them for real use, retrain on real data (same pipeline):

```bash
python make_training_data.py                                   # (synthetic) or supply dataset/email_dataset.csv: text,label
python train_email_model.py                                    # -> email_model.pkl
python train_pdf_model.py --csv dataset/pdf_features.csv       # or --safe-dir/--malicious-dir with real PDFs

# after changing make_training_data.py, rebuild + retrain the email model:
python make_training_data.py --only email
python train_email_model.py

# run the regression tests (includes the SPAM tests):
python -m unittest test_email_pdf_accuracy test_spam_detection
python evaluate_models.py                                      # honest report
python -m unittest test_ml_models -v
```
