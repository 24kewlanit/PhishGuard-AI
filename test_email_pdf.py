"""
test_email_pdf.py
=================
Automated checks for the Email + PDF scanner. Runs everything in memory with
Flask's test client - no server needs to be running, nothing is executed.

Run:   py -3.11 test_email_pdf.py

It verifies:
  * the existing endpoints (/health, /history, /) still work
  * email-only, PDF-only and combined scans
  * error handling (bad extension, fake PDF, empty, too large)
  * the temporary PDF folder is EMPTY after every request (cleanup works),
    including when the analyzer crashes
"""

import io
import os
import sys

import make_sample_files as S
import email_pdf_routes as R
from app import app

client = app.test_client()
failures = []


def check(name, condition, detail=""):
    print(("  PASS  " if condition else "  FAIL  ") + name + (f"   {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def scan(**fields):
    data = {}
    for key, value in fields.items():
        data[key] = value
    resp = client.post("/email-pdf-scan", data=data, content_type="multipart/form-data")
    return resp.status_code, resp.get_json()


def f(content, name):
    return (io.BytesIO(content if isinstance(content, bytes) else content.encode()), name)


def temp_is_empty():
    return os.listdir(R.PDF_TEMP_DIR) == []


print("\n[1] Existing project still works")
check("GET /health", client.get("/health").get_json()["status"] == "online")
check("GET /history", client.get("/history").get_json()["success"] is True)
check("GET / (page loads, URL scanner present)",
      b'id="urlInput"' in client.get("/").data and b'id="emailPdfScanner"' in client.get("/").data)

print("\n[2] Email only")
st, j = scan(email_file=f(S.EMAILS["safe_email.eml"], "safe.eml"))
check("safe email -> SAFE", j["overall"]["classification"] == "SAFE", j["overall"])
st, j = scan(email_file=f(S.EMAILS["spam_email.eml"], "spam.eml"))
check("spam email -> SPAM", j["overall"]["classification"] == "SPAM", j["overall"])
st, j = scan(email_file=f(S.EMAILS["phishing_email.eml"], "phish.eml"))
check("phishing email -> PHISHING", j["overall"]["classification"] == "PHISHING", j["overall"])
check("phishing .eml: URLs extracted", len(j["email"]["urls"]) >= 1)
check("phishing .eml: reasons present", len(j["email"]["reasons"]) >= 3)
# Updated for the accuracy fix: a bare pasted message has no sender / headers, so it can NOT be
# called SAFE - it is INCONCLUSIVE (and has no fake 0/100 score).
st, j = scan(email_text="Hi, are we still on for lunch tomorrow at 1?")
check("pasted body only (no sender) -> INCONCLUSIVE, not SAFE", j["email"]["classification"] == "INCONCLUSIVE", j["email"])
check("pasted body only: no risk score invented", j["email"]["risk_score"] is None)
st, j = scan(email_text="From: Anita <anita@company-example.org>\nSubject: Lunch\n\nHi, are we still on for lunch tomorrow at 1?")
check("pasted benign email WITH headers -> SAFE", j["email"]["classification"] == "SAFE", j["email"])
check("pasted benign email: no invented reasons", j["email"]["reasons"] == [])

print("\n[3] PDF only")
st, j = scan(pdf_file=f(S.normal_pdf(), "normal.pdf"))
check("normal PDF -> SAFE", j["pdf"]["classification"] == "SAFE", j["pdf"])
st, j = scan(pdf_file=f(S.javascript_only_pdf(), "js.pdf"))
check("JavaScript PDF -> SUSPICIOUS", j["pdf"]["classification"] == "SUSPICIOUS")
check("JavaScript flag set", j["pdf"]["javascript"] is True)
st, j = scan(pdf_file=f(S.embedded_file_pdf(), "emb.pdf"))
check("embedded-file PDF -> not SAFE", j["pdf"]["classification"] != "SAFE")
st, j = scan(pdf_file=f(S.malicious_indicators_pdf(), "bad.pdf"))
check("OpenAction+JS+Launch PDF -> POTENTIALLY MALICIOUS",
      j["pdf"]["classification"] == "POTENTIALLY MALICIOUS")

print("\n[4] Combined logic (overrides)")
st, j = scan(email_file=f(S.EMAILS["safe_email.eml"], "s.eml"), pdf_file=f(S.malicious_indicators_pdf(), "b.pdf"))
check("safe email + malicious PDF is NOT safe", j["overall"]["classification"] != "SAFE", j["overall"])
check("override explanation shown", any(r.startswith("Override") for r in j["overall"]["reasons"]))
st, j = scan(email_file=f(S.EMAILS["phishing_email.eml"], "p.eml"), pdf_file=f(S.normal_pdf(), "n.pdf"))
check("phishing email + normal PDF stays PHISHING", j["overall"]["classification"] == "PHISHING")
st, j = scan(email_file=f(S.EMAILS["safe_email.eml"], "s.eml"), pdf_file=f(S.normal_pdf(), "n.pdf"))
check("safe email + normal PDF -> SAFE", j["overall"]["classification"] == "SAFE")
st, j = scan(email_file=f(S.EMAILS["spam_email.eml"], "s.eml"), pdf_file=f(S.normal_pdf(), "n.pdf"))
check("spam email + normal PDF -> SPAM", j["overall"]["classification"] == "SPAM")

print("\n[5] Error handling")
st, j = scan()
check("nothing uploaded -> 400", st == 400 and j["success"] is False)
st, j = scan(pdf_file=f(b"MZ not a pdf", "evil.pdf"))
check("fake PDF (bad magic bytes) -> 400", st == 400)
st, j = scan(pdf_file=f(S.normal_pdf(), "notes.txt"))
check("wrong PDF extension -> 400", st == 400)
st, j = scan(email_file=f("hello", "mail.txt"))
check("wrong email extension -> 400", st == 400)
st, j = scan(pdf_file=f(b"", "empty.pdf"))
check("empty PDF -> 400", st == 400)
st, j = scan(pdf_file=f(b"%PDF-" + b"0" * (R.MAX_PDF_BYTES + 10), "big.pdf"))
check("oversized PDF -> 413", st == 413)
st, j = scan(email_text="a" * (R.MAX_EMAIL_TEXT_CHARS + 1))
check("oversized pasted text -> 413", st == 413)
st, j = scan(pdf_file=f(S.normal_pdf(), "../../etc/passwd<script>.pdf"))
check("hostile filename is sanitised", "/" not in j["pdf"]["filename"] and "<" not in j["pdf"]["filename"], j["pdf"]["filename"])

print("\n[6] Temporary file cleanup")
check("temp folder empty after all the scans above", temp_is_empty(), str(os.listdir(R.PDF_TEMP_DIR)))
original = R.analyze_pdf


def crash(path):
    assert os.path.exists(path)          # the temp copy really existed during analysis
    raise RuntimeError("secret internal detail /server/path")


R.analyze_pdf = crash
st, j = scan(pdf_file=f(S.normal_pdf(), "n.pdf"))
R.analyze_pdf = original
check("analyzer crash -> 500 with generic message", st == 500 and "secret" not in str(j) and "/server" not in str(j))
check("temp folder empty even after a crash", temp_is_empty())

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASSED")
