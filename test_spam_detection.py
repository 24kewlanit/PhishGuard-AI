"""
test_spam_detection.py
======================
Regression tests for the SPAM fix (promotional / marketing / lottery / casino mail and PDFs).

Run:   python -m unittest test_spam_detection -v

Background: real-world promotional mail (prices, "% off", plan tiers, "Learn more", an
unsubscribe footer) used to be reported SAFE by the e-mail scanner, and spam PDFs were
always SAFE because the PDF scanner had no SPAM verdict. These tests keep both fixed,
and also make sure ordinary legitimate mail (invoices, receipts, renewals, newsletters)
does NOT become SPAM.

No real e-mail addresses or files are used. Everything is built in memory.
"""

import io
import os
import tempfile
import unittest

import email_pdf_fixtures as F
from app import app
from combined_analyzer import combine_results
from email_analyzer import parse_eml_bytes, parse_email_text
from email_model import load_email_model, predict_email
from pdf_analyzer import analyze_pdf

MODEL = load_email_model()
client = app.test_client()


def eml(sender, subject, body, extra_headers=""):
    return (f"From: {sender}\nTo: reader@example.org\nSubject: {subject}\n"
            f"Date: Wed, 30 Sep 2026 11:10:34 +0000\nMessage-Id: <1@example.com>\n"
            f"{extra_headers}MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\n\n{body}\n").encode()


def classify(raw, model=MODEL):
    return predict_email(parse_eml_bytes(raw), model)


def html_marketing_eml():
    """A polished, authenticated marketing e-mail (HTML, many links, bulk-mail headers)."""
    links = "".join(f'<a href="https://track.example.net/c/{i}">Learn More &gt;&gt;</a><br>' for i in range(12))
    html = ("<html><body><h1>Spring Plan Sale: Up to 30% Off</h1>"
            "<p>Annual plan $79</p><p>Lifetime plan $149</p><p>Team bundle $249</p>"
            "<p>Free upgrades and priority support.</p>" + links +
            "<p>You are receiving this e-mail because you signed up to receive newsletters. "
            'If you do not want to receive this type of e-mail, please click to <a href="https://x.example.net/u">unsubscribe</a>.</p>'
            "</body></html>")
    headers = ("From: Acme Design <hello@mail.acmedesign.example>\nTo: reader@example.org\n"
               "Subject: Spring Plan Sale: Up to 30% Off\nDate: Wed, 30 Sep 2026 11:10:34 +0000\n"
               "Message-Id: <2@acmedesign.example>\nMIME-Version: 1.0\nX-Mailgun-Track: true\nX-Mailgun-Tag: 123\n"
               "Authentication-Results: mx.example.org; dkim=pass header.i=@mail.acmedesign.example; "
               "spf=pass smtp.mailfrom=mail.acmedesign.example; dmarc=pass header.from=acmedesign.example\n"
               'Content-Type: text/html; charset="utf-8"\nContent-Transfer-Encoding: 8bit\n\n')
    return (headers + html).encode()


class EmailSpam(unittest.TestCase):
    def test_html_marketing_email_is_spam_by_rules_alone(self):
        r = classify(html_marketing_eml(), model=None)
        self.assertEqual(r["classification"], "SPAM", r["reasons"])

    def test_html_marketing_email_is_spam_with_ml(self):
        r = classify(html_marketing_eml())
        self.assertEqual(r["classification"], "SPAM", r["reasons"])

    def test_authenticated_marketing_is_spam_but_not_alarming(self):
        r = classify(html_marketing_eml(), model=None)
        self.assertLessEqual(r["risk_score"], 40)
        self.assertTrue(any("authentication" in n for n in r["notes"]), r["notes"])

    def test_reasons_are_specific_and_real(self):
        text = " ".join(classify(html_marketing_eml(), model=None)["reasons"]).lower()
        for expected in ("promotional", "percentage", "mass-mailing", "bulk-mailing"):
            self.assertIn(expected, text)

    def test_plain_text_promo_without_headers(self):
        raw = ("Subject: FLASH SALE - 70% OFF everything\n\nUp to 70% off shoes and bags. Free shipping over $50. "
               "Promo code SAVE70. Hurry, ends tonight! Shop now. Unsubscribe | Manage preferences")
        r = predict_email(parse_email_text(raw), MODEL)
        self.assertEqual(r["classification"], "SPAM", r["reasons"])

    def test_lottery_and_casino_still_spam(self):
        for subj, body in (("YOU WON", "Congratulations! You have won $1,000,000 in the lottery. Claim your prize, no obligation."),
                           ("Win big", "Play online casino games, claim your free spins now. Guaranteed returns, risk free!!!")):
            r = classify(eml("Prize <win@x.example>", subj, body))
            self.assertEqual(r["classification"], "SPAM", (subj, r["reasons"]))

    def test_spam_never_hides_phishing(self):
        raw = eml("PayPal <service@paypa1-secure.top>", "Sale! Action required",
                  "50% off sale. Your account will be suspended within 24 hours. Verify your account password now: "
                  "http://paypa1-secure.top/login")
        self.assertIn(classify(raw)["classification"], ("SUSPICIOUS", "PHISHING"))


class LegitMailStaysSafe(unittest.TestCase):
    CASES = {
        "order shipped, several prices": ("Orders <orders@shop.example>", "Your order has shipped",
            "Item: USB cable $9.99. Shipping $4.50. Tax $1.20. Total $15.69. Track your package. Delivery Thursday."),
        "invoice": ("Accounts <billing@acme-corp.example>", "Invoice 2041",
            "Consulting 10 hours at $90 = $900. VAT $180. Total $1,080. Payment due within 30 days."),
        "subscription renewal": ("Billing <billing@streaming.example>", "Your subscription renews soon",
            "Your Premium plan renews on 12 Nov for $11.99. Manage or cancel any time in account settings."),
        "informational newsletter": ("News <news@python.example.org>", "This week in Python",
            "Top stories: release notes, three tutorials and a meetup on Saturday. You are receiving this newsletter "
            "because you subscribed. Unsubscribe at any time."),
        "job offer letter": ("HR <hr@acme-corp.example>", "Your offer letter",
            "We are pleased to offer you the Senior Engineer position. Salary $95,000 per year. Please sign by Friday."),
        "work mail": ("Priya <priya@acme-corp.example>", "Meeting moved",
            "The sprint review moves to 3pm Friday. The $5,000 offsite budget is approved."),
        "personal": ("Mom <mom@gmail.com>", "Dinner", "Can you come on Sunday? There is a sale at the market, I will get vegetables."),
    }

    def test_legit_mail_is_not_spam(self):
        for name, (sender, subject, body) in self.CASES.items():
            with self.subTest(name):
                r = classify(eml(sender, subject, body))
                self.assertEqual(r["classification"], "SAFE", (name, r["classification"], r["reasons"]))


def _pdf(lines, links=()):
    data = F.text_pdf(lines, links=links)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


class PdfSpam(unittest.TestCase):
    def _run(self, lines, links=()):
        path = _pdf(lines, links)
        try:
            return analyze_pdf(path)
        finally:
            os.remove(path)

    def test_lottery_pdf_is_spam(self):
        r = self._run(["CONGRATULATIONS! You have won $1,500,000 in the international lottery.",
                       "You are the lucky winner. Claim your free prize today, no obligation, risk free.",
                       "Earn money from home, work from home with no experience. Limited offer, act now!"])
        self.assertEqual(r["classification"], "SPAM", r["reasons"])

    def test_sale_flyer_pdf_is_spam(self):
        r = self._run(["MEGA SALE - Up to 70% OFF everything!", "Lifetime deal: now $199, was $499. Limited time offer.",
                       "Exclusive discount for subscribers. Shop now. Use promo code SAVE70. Free shipping.",
                       "To unsubscribe from our newsletter click here."])
        self.assertEqual(r["classification"], "SPAM", r["reasons"])
        self.assertIsNotNone(r["risk_score"])

    def test_ordinary_pdfs_stay_safe(self):
        for lines in (["Quarterly Engineering Report - Q3", "The team shipped 14 features and closed 92 tickets.",
                       "Performance improved by 12 percent after the database migration."],
                      ["INVOICE 2041 - Acme Corp Ltd", "Consulting, 10 hours at $90/hour. Total due: $900.",
                       "Payment due within 30 days. Bank account details: IBAN GB00TEST123."]):
            r = self._run(lines)
            self.assertEqual(r["classification"], "SAFE", r["reasons"])

    def test_dangerous_pdf_is_never_downgraded_to_spam(self):
        # Sales wording on top of a phishing lure + look-alike link must stay a security verdict.
        r = self._run(["PayPal: your account will be suspended within 24 hours. Verify your account password now.",
                       "50% off sale, shop now, limited time offer!"],
                      links=["http://paypa1-secure-login.top/verify"])
        self.assertIn(r["classification"], ("SUSPICIOUS", "POTENTIALLY MALICIOUS"), r["reasons"])


class CombinedAndApi(unittest.TestCase):
    E = staticmethod(lambda c, r: {"classification": c, "risk_score": r, "reasons": []})

    def test_spam_pdf_in_safe_email_is_not_safe(self):
        r = combine_results(self.E("SAFE", 3), self.E("SPAM", 49))
        self.assertEqual(r["classification"], "SPAM")
        self.assertGreaterEqual(r["risk_score"], 25)

    def test_pdf_only_spam(self):
        self.assertEqual(combine_results(None, self.E("SPAM", 49))["classification"], "SPAM")

    def test_malicious_pdf_still_beats_spam_email(self):
        r = combine_results(self.E("SPAM", 40), self.E("POTENTIALLY MALICIOUS", 90))
        self.assertEqual(r["classification"], "PHISHING")

    def test_api_returns_spam_for_marketing_eml_upload(self):
        resp = client.post("/email-pdf-scan", data={"email_file": (io.BytesIO(html_marketing_eml()), "promo.eml")},
                           content_type="multipart/form-data")
        body = resp.get_json()
        self.assertTrue(body["success"], body)
        self.assertEqual(body["email"]["classification"], "SPAM")
        self.assertEqual(body["overall"]["classification"], "SPAM")

    def test_api_returns_spam_for_spam_pdf_upload(self):
        pdf = F.text_pdf(["CONGRATULATIONS! You have won the lottery prize of $1,000,000.",
                          "You are the lucky winner. Claim your free prize now, no obligation, risk free!!!"])
        resp = client.post("/email-pdf-scan", data={"pdf_file": (io.BytesIO(pdf), "prize.pdf")},
                           content_type="multipart/form-data")
        body = resp.get_json()
        self.assertTrue(body["success"], body)
        self.assertEqual(body["pdf"]["classification"], "SPAM")
        self.assertEqual(body["overall"]["classification"], "SPAM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
