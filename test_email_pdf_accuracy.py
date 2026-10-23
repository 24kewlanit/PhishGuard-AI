"""
test_email_pdf_accuracy.py
==========================
Regression + accuracy tests for Email Detection and PDF Detection.

Run:   python -m unittest test_email_pdf_accuracy -v

Everything runs in memory with Flask's test client. Fixtures are generated
from content (email_pdf_fixtures.py); results are never chosen from file
names. Nothing is executed, opened or visited.

What is verified for every case: the classification (or an appropriate
INCONCLUSIVE), the risk score, the reasons shown to the user, the API
response, and that failed/partial analysis is never reported SAFE.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import email_pdf_fixtures as F
import email_pdf_routes as R
import email_analyzer
import pdf_analyzer
from app import app
from combined_analyzer import combine_results
from email_analyzer import parse_eml_bytes, parse_email_text
from email_model import predict_email
from pdf_analyzer import analyze_pdf

client = app.test_client()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def scan(**fields):
    resp = client.post("/email-pdf-scan", data=fields, content_type="multipart/form-data")
    return resp.status_code, resp.get_json()


def upload(content, name):
    return io.BytesIO(content if isinstance(content, bytes) else content.encode("utf-8")), name


def eml(text):
    return predict_email(parse_eml_bytes(text.encode("utf-8")))


def pdf(data):
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(fd, data)
        os.close(fd)
        return analyze_pdf(path)
    finally:
        os.remove(path)


def reasons_text(result):
    return " | ".join(result["reasons"]).lower()


NOT_SAFE = {"SUSPICIOUS", "PHISHING", "POTENTIALLY MALICIOUS", "SPAM", "INCONCLUSIVE"}


# ===========================================================================
# EMAIL
# ===========================================================================
class EmailDetection(unittest.TestCase):

    # 1
    def test_01_normal_email_is_safe_with_no_false_reasons(self):
        r = eml(F.EML_NORMAL)
        self.assertEqual(r["classification"], "SAFE")
        self.assertEqual(r["analysis_status"], "complete")
        self.assertEqual(r["reasons"], [])
        self.assertIsInstance(r["risk_score"], int)
        self.assertLess(r["risk_score"], 25)
        self.assertTrue(r["checks_performed"])           # user can see what was checked

    # 2
    def test_02_urgent_password_phishing(self):
        r = eml(F.EML_PHISHING_PASSWORD)
        self.assertEqual(r["classification"], "PHISHING")
        self.assertGreaterEqual(r["risk_score"], 60)
        text = reasons_text(r)
        self.assertIn("credential", text)
        self.assertIn("urgent", text)

    def test_02b_password_plus_urgency_without_any_link_is_not_safe(self):
        # root-cause regression: this used to score 16 -> SAFE
        r = eml("From: IT <it@acme-corp.example>\nSubject: Notice\n\n"
                "Your password has expired. Act immediately and confirm your password.")
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("combination", reasons_text(r))

    def test_02c_single_weak_keyword_is_not_enough(self):
        r = eml("From: IT <it@acme-corp.example>\nSubject: Tips\n\n"
                "Reminder: choose a long password for the new laptop, details are on the wiki page.")
        self.assertEqual(r["classification"], "SAFE")

    # 3
    def test_03_misleading_url(self):
        r = eml(F.EML_MISLEADING_LINK)
        self.assertIn(r["classification"], {"SUSPICIOUS", "PHISHING"})
        self.assertIn("misleading link", reasons_text(r))

    def test_03b_lookalike_domain_typed_without_scheme(self):
        r = predict_email(parse_email_text("From: Support <help@acme-corp.example>\nSubject: Account\n\n"
                                           "Please sign in at paypa1-secure.com to keep your account active."))
        self.assertIn(r["classification"], {"SUSPICIOUS", "PHISHING"})
        self.assertIn("paypa1-secure.com", " ".join(r["reasons"]))

    def test_03c_lookalike_detector_unit(self):
        from email_model import lookalike_brand
        for host, brand in [("paypa1.com", "paypal"), ("micr0soft.com", "microsoft"),
                            ("arnazon.com", "amazon"), ("gogle.com", "google"),
                            ("g00gle-login.com", "google"), ("netfl1x.com", "netflix")]:
            self.assertEqual(lookalike_brand(host), brand, host)
        for host in ("paypal.com", "www.google.com", "docs.google.com", "amazon.co.uk",
                     "example.com", "purchase.com", "pineapple.org", "apply.com", "github.com"):
            self.assertIsNone(lookalike_brand(host), host)

    # 4
    def test_04_sender_replyto_mismatch(self):
        r = eml(F.EML_REPLYTO_MISMATCH)
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("reply-to", reasons_text(r))

    def test_04b_related_replyto_domain_is_not_flagged(self):
        r = eml(F.EML_NORMAL.replace("Subject:", "Reply-To: priya@mail.acme-corp.example\nSubject:", 1))
        self.assertEqual(r["classification"], "SAFE")

    # 5
    def test_05_trusted_organisation_impersonation(self):
        r = eml(F.EML_IMPERSONATION)
        self.assertIn(r["classification"], {"SUSPICIOUS", "PHISHING"})
        self.assertIn("microsoft", reasons_text(r))

    def test_05b_real_brand_domain_is_not_impersonation(self):
        r = eml(F.LEGIT_EMAILS[6])         # genuine accounts.google.com sender
        self.assertEqual(r["classification"], "SAFE", r["reasons"])

    # 6
    def test_06_missing_headers_is_inconclusive_not_safe_not_malicious(self):
        for r in (predict_email(parse_email_text("Please review the quarterly report when you have a "
                                                 "moment and send me your comments.")),
                  eml(F.EML_NO_FROM)):
            self.assertEqual(r["classification"], "INCONCLUSIVE")
            self.assertIsNone(r["risk_score"])            # no fake 0/100
            self.assertEqual(r["analysis_status"], "partial")
            self.assertIn("sender", reasons_text(r))

    def test_06b_missing_auth_headers_alone_do_not_block_a_verdict(self):
        r = eml(F.EML_NO_AUTH_HEADERS)
        self.assertEqual(r["classification"], "SAFE")
        self.assertTrue(any("SPF/DKIM/DMARC" in n for n in r["notes"]))

    def test_06c_failed_authentication_is_evidence(self):
        r = eml(F.PHISHING_EMAILS[-2])
        self.assertIn("authentication failed", reasons_text(r))
        self.assertNotEqual(r["classification"], "SAFE")

    def test_06d_passing_authentication_does_not_hide_other_evidence(self):
        text = F.EML_NORMAL.replace("Thanks,", "Verify your password immediately at http://203.0.113.7/x\nThanks,")
        self.assertNotEqual(eml(text)["classification"], "SAFE")

    # 7
    def test_07_empty_and_malformed_input(self):
        st, j = scan(email_text="   \n  ")
        self.assertEqual(st, 400)
        st, j = scan(email_file=upload(b"", "e.eml"))
        self.assertEqual(st, 400)
        st, j = scan(email_file=upload(b"   \n\n  ", "e.eml"))
        self.assertEqual(st, 400)
        # unreadable binary junk is analysed and reported INCONCLUSIVE (never SAFE)
        junk = bytes(range(0, 32)) * 10 + b"\xff\xfe\x00\x01" * 20
        st, j = scan(email_file=upload(junk, "junk.eml"))
        self.assertEqual(st, 200)
        self.assertEqual(j["email"]["classification"], "INCONCLUSIVE")
        self.assertEqual(j["overall"]["classification"], "INCONCLUSIVE")
        self.assertIsNone(j["overall"]["risk_score"])
        st, j = scan(email_text="\x00\x01\x02\x03\x04\x05\x06\x07\x08 \x0b\x0c\x0e\x0f " * 5)
        self.assertEqual(j["email"]["classification"], "INCONCLUSIVE")

    def test_07b_too_little_content(self):
        r = eml("From: a@acme-corp.example\nSubject: Hi\n\n.")
        self.assertEqual(r["classification"], "INCONCLUSIVE")

    # 8
    def test_08_legitimate_and_suspicious_indicators_together(self):
        r = eml(F.EML_MIXED)
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertLess(r["risk_score"], 60)               # not over-claimed as phishing
        self.assertIn("shortener", reasons_text(r))
        self.assertIn("spf=pass", " ".join(r["checks_performed"]))   # the legitimate side is shown too

    # 9
    def test_09_unparseable_email_never_becomes_safe(self):
        with mock.patch.object(email_analyzer, "message_from_bytes", side_effect=RuntimeError("boom")):
            parsed = parse_eml_bytes(F.EML_NORMAL.encode())
        self.assertFalse(parsed["parse_ok"])
        r = predict_email(parsed)
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertEqual(r["analysis_status"], "failed")
        # through the API as well
        with mock.patch.object(email_analyzer, "message_from_bytes", side_effect=RuntimeError("boom")):
            st, j = scan(email_file=upload(F.EML_NORMAL, "a.eml"))
        self.assertEqual(st, 200)
        self.assertEqual(j["email"]["classification"], "INCONCLUSIVE")
        self.assertNotIn("boom", json.dumps(j))             # no internal details leak

    def test_09b_parse_failure_of_pasted_text(self):
        with mock.patch.object(email_analyzer, "message_from_string", side_effect=ValueError("x")):
            parsed = parse_email_text("From: a@b.example\nSubject: s\n\nbody text here")
        self.assertEqual(predict_email(parsed)["classification"], "INCONCLUSIVE")

    # 10
    def test_10_suspicious_attachment(self):
        r = eml(F.EML_ATTACHMENT_EXE)
        self.assertIn(r["classification"], {"SUSPICIOUS", "PHISHING"})
        self.assertIn("double extension", reasons_text(r))
        self.assertIn("executable", reasons_text(r))

    def test_10b_attachment_only_email_is_analysed_not_rejected(self):
        st, j = scan(email_file=upload(F.EML_ATTACHMENT_ONLY, "a.eml"))
        self.assertEqual(st, 200)
        self.assertNotEqual(j["email"]["classification"], "SAFE")
        self.assertIn("update.scr", json.dumps(j["email"]["reasons"]))

    def test_10c_text_reference_to_executable(self):
        r = eml(F.EML_TEXT_REFERENCE_ONLY)
        self.assertNotEqual(r["classification"], "SAFE")
        self.assertIn("security-update.exe", reasons_text(r))

    # ----- extra accuracy checks -----
    def test_html_part_cannot_hide_phishing_behind_harmless_text_part(self):
        raw = ("From: A <a@acme-corp.example>\nTo: b@example.org\nSubject: Hello\nMIME-Version: 1.0\n"
               'Content-Type: multipart/alternative; boundary="X"\n\n--X\nContent-Type: text/plain\n\n'
               "Hello, see you at the meeting tomorrow morning.\n--X\nContent-Type: text/html\n\n"
               "<p>Verify your account password immediately: "
               '<a href="http://203.0.113.5/login">http://acme-corp.example</a></p>\n--X--\n')
        r = eml(raw)
        self.assertNotEqual(r["classification"], "SAFE")

    def test_false_positive_rate_on_legitimate_corpus(self):
        wrong = [(i, eml(t)["classification"]) for i, t in enumerate(F.LEGIT_EMAILS)
                 if eml(t)["classification"] != "SAFE"]
        self.assertEqual(wrong, [], f"legitimate emails not SAFE: {wrong}")

    def test_no_phishing_sample_is_ever_safe(self):
        bad = [(i, eml(t)["classification"]) for i, t in enumerate(F.PHISHING_EMAILS)
               if eml(t)["classification"] in ("SAFE", "SPAM")]
        self.assertEqual(bad, [], f"missed phishing samples: {bad}")

    def test_result_does_not_depend_on_file_name(self):
        names = ["safe.eml", "phishing.eml", "invoice_urgent_verify_password.eml", "a.eml"]
        for content, expected in ((F.EML_NORMAL, "SAFE"), (F.EML_PHISHING_PASSWORD, "PHISHING")):
            for name in names:
                st, j = scan(email_file=upload(content, name))
                self.assertEqual(j["email"]["classification"], expected, name)

    def test_spam_still_spam(self):
        r = eml("From: Deals <deals@bulkmail.example>\nSubject: WINNER!!! FREE prize\n\n"
                "You are the lucky winner of our lottery! Claim your free gift now, 100% free, no obligation, "
                "limited offer!!! Earn money fast, work from home, buy now!")
        self.assertEqual(r["classification"], "SPAM")

    def test_ml_path_cannot_hide_rule_findings_or_incompleteness(self):
        class FakeModel:
            classes_ = ["phishing", "safe", "spam"]

            def predict_proba(self, X):
                return [[0.01, 0.98, 0.01]]            # model says "safe"
        r = predict_email(parse_eml_bytes(F.EML_PHISHING_PASSWORD.encode()), FakeModel())
        self.assertEqual(r["classification"], "PHISHING")
        r = predict_email(parse_email_text("Please review the quarterly report when you can."), FakeModel())
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        r = predict_email(parse_eml_bytes(F.EML_NORMAL.encode()), FakeModel())
        self.assertEqual(r["classification"], "SAFE")
        self.assertTrue(r["ml_used"])


# ===========================================================================
# PDF
# ===========================================================================
class PdfDetection(unittest.TestCase):

    # 1
    def test_01_ordinary_pdf_is_safe(self):
        for compress in (None, "flate", "a85flate"):
            r = pdf(F.text_pdf(F.NORMAL_TEXT, compress))
            self.assertEqual(r["classification"], "SAFE", compress)
            self.assertEqual(r["analysis_status"], "complete")
            self.assertEqual(r["risk_score"], 0)
            self.assertGreater(r["text_chars"], 200)            # text was really read
            self.assertTrue(any("Text extraction" in c for c in r["checks_performed"]))

    # 2
    def test_02_phishing_text_pdf(self):
        r = pdf(F.text_pdf(F.PHISHING_TEXT, "flate"))
        self.assertEqual(r["classification"], "POTENTIALLY MALICIOUS")
        text = reasons_text(r)
        for expected in ("credential", "urgent", "wire transfer"):
            self.assertIn(expected, text)

    def test_02b_ordinary_invoice_with_bank_details_is_not_flagged(self):
        r = pdf(F.text_pdf(["Invoice 2041 - October consulting services",
                            "Amount due: 1,200.00. Payment terms: 30 days from the invoice date.",
                            "Bank account: 12345678, IBAN: GB00TEST00000012345678, SWIFT code TESTGB2L",
                            "Thank you for your business."]))
        self.assertEqual(r["classification"], "SAFE", r["reasons"])

    # 3
    def test_03_suspicious_external_urls(self):
        r = pdf(F.text_pdf(F.NORMAL_TEXT, links=["http://203.0.113.9/login"]))
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("raw ip address", reasons_text(r))
        r = pdf(F.text_pdf(F.NORMAL_TEXT + ["Download here: http://203.0.113.77/files/report"]))
        self.assertIn("raw ip address", reasons_text(r))            # URL only in the text, no annotation
        self.assertNotEqual(r["classification"], "SAFE")

    def test_03b_plain_external_link_alone_is_not_malicious(self):
        r = pdf(F.text_pdf(F.NORMAL_TEXT, links=["https://www.example.com/policy"]))
        self.assertEqual(r["classification"], "SAFE")
        self.assertIn("external link", reasons_text(r))             # still reported to the user

    # 4
    def test_04_lookalike_domain(self):
        r = pdf(F.text_pdf(F.NORMAL_TEXT, links=["http://paypa1.com/verify"]))
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("look-alike", reasons_text(r))
        r = pdf(F.text_pdf(F.NORMAL_TEXT + ["Visit micr0soft-support.com for help"]))
        self.assertNotEqual(r["classification"], "SAFE")

    # 5
    def test_05_javascript_and_actions(self):
        import make_sample_files as S
        r = pdf(S.malicious_indicators_pdf())
        self.assertEqual(r["classification"], "POTENTIALLY MALICIOUS")
        self.assertTrue(r["javascript"])
        r = pdf(S.javascript_only_pdf())
        self.assertEqual(r["classification"], "SUSPICIOUS")           # JS alone: review, not malware
        r = pdf(S.embedded_file_pdf())
        self.assertNotEqual(r["classification"], "SAFE")

    def test_05b_javascript_hidden_in_compressed_object_stream(self):
        r = pdf(F.objstm_javascript_pdf())
        self.assertTrue(r["javascript"])
        self.assertIn(r["classification"], {"SUSPICIOUS", "POTENTIALLY MALICIOUS"})

    def test_05c_obfuscated_keyword(self):
        r = pdf(F.obfuscated_js_pdf())
        self.assertIn("obfuscated", reasons_text(r))
        self.assertEqual(r["classification"], "POTENTIALLY MALICIOUS")

    def test_05d_form_asking_for_a_password(self):
        r = pdf(F.widget_form_pdf(["Full name", "Password"]))
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("sensitive", reasons_text(r))
        r = pdf(F.widget_form_pdf(["First name", "Comments"]))
        self.assertEqual(r["classification"], "SAFE")                 # harmless form is not flagged as dangerous

    # 6
    def test_06_empty_or_corrupted(self):
        st, j = scan(pdf_file=upload(b"", "e.pdf"))
        self.assertEqual(st, 400)
        for label, data in (("header only", b"%PDF-1.4\n"),
                            ("garbage", b"%PDF-1.4\nthis is not a real pdf \x00\x01\x02 garbage"),
                            ("truncated", F.text_pdf(F.NORMAL_TEXT)[:150])):
            st, j = scan(pdf_file=upload(data, "x.pdf"))
            self.assertEqual(st, 200, label)
            self.assertEqual(j["pdf"]["classification"], "INCONCLUSIVE", label)
            self.assertIsNone(j["pdf"]["risk_score"], label)
            self.assertIn(j["pdf"]["analysis_status"], {"failed", "partial"}, label)
            self.assertEqual(j["overall"]["classification"], "INCONCLUSIVE", label)

    # 7
    def test_07_encrypted_password_protected(self):
        r = pdf(F.encrypted_pdf(F.NORMAL_TEXT))
        self.assertEqual(r["classification"], "INCONCLUSIVE")         # not SAFE, and not "malicious" either
        self.assertTrue(r["encrypted"])
        self.assertIn("encrypted", reasons_text(r))
        self.assertIsNone(r["risk_score"])

    # 8
    def test_08_image_only_pdf_without_ocr(self):
        r = pdf(F.image_only_pdf())
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIn("ocr is not available", reasons_text(r))
        self.assertEqual(r["text_extraction"], "none")

    # 9
    def test_09_extraction_errors(self):
        # damaged compressed stream
        r = pdf(F.corrupt_flate_pdf())
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        # text that cannot be decoded without PyMuPDF
        if not pdf_analyzer.PYMUPDF_AVAILABLE:
            r = pdf(F.hex_glyph_text_pdf())
            self.assertEqual(r["classification"], "INCONCLUSIVE")
        # the scanner itself crashes -> failed analysis, never SAFE
        with mock.patch.object(pdf_analyzer, "_raw_scan", side_effect=OSError("disk")):
            r = pdf(F.text_pdf(F.NORMAL_TEXT))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertEqual(r["analysis_status"], "failed")
        with mock.patch.object(pdf_analyzer, "extract_pdf_findings", side_effect=RuntimeError("x")):
            r = pdf(F.text_pdf(F.NORMAL_TEXT))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIn("failed unexpectedly", reasons_text(r))

    # 10
    def test_10_normal_content_with_suspicious_indicators(self):
        r = pdf(F.text_pdf(F.MIXED_TEXT))
        self.assertIn(r["classification"], {"SUSPICIOUS", "POTENTIALLY MALICIOUS"})
        text = reasons_text(r)
        self.assertIn("look-alike", text)
        self.assertIn("credential", text)

    # ----- extra -----
    def test_text_extractor_unit(self):
        ex = pdf_analyzer._extract_content_text
        text, bad = ex(b"BT (Hello \\(World\\)) Tj [(Ver)-20(ify) -400 (now)] TJ <0041004200430044> Tj ET")
        self.assertIn("Hello (World)", text)
        self.assertIn("Verify now", text)
        self.assertIn("ABCD", text)
        self.assertEqual(bad, 0)
        text, bad = ex(b"BT <00030004000500060007> Tj ET")
        self.assertEqual(text.strip(), "")
        self.assertGreater(bad, 0)
        text, _ = ex(b"BT (caf\\351 \\101) Tj ET")
        self.assertIn("A", text)

    def test_no_pdf_in_the_corpus_is_hidden_by_compression(self):
        a = pdf(F.text_pdf(F.PHISHING_TEXT))
        b = pdf(F.text_pdf(F.PHISHING_TEXT, "flate"))
        c = pdf(F.text_pdf(F.PHISHING_TEXT, "a85flate"))
        self.assertEqual({a["classification"], b["classification"], c["classification"]},
                         {"POTENTIALLY MALICIOUS"})

    def test_result_does_not_depend_on_file_name(self):
        for name in ("report.pdf", "SAFE_document.pdf", "malware_invoice.pdf", "z.pdf"):
            st, j = scan(pdf_file=upload(F.text_pdf(F.NORMAL_TEXT), name))
            self.assertEqual(j["pdf"]["classification"], "SAFE", name)
            st, j = scan(pdf_file=upload(F.text_pdf(F.PHISHING_TEXT), name))
            self.assertEqual(j["pdf"]["classification"], "POTENTIALLY MALICIOUS", name)

    def test_pdf_with_no_headers_junk_before_header(self):
        data = b"JUNKJUNK" * 4 + F.text_pdf(F.NORMAL_TEXT)
        r = pdf(data)
        self.assertIn("junk data before", reasons_text(r))


class PdfWithFakePyMuPDF(unittest.TestCase):
    """
    PyMuPDF is not installed in every environment (it is in requirements.txt).
    These tests drive the PyMuPDF code path with a FAKE module so that the
    integration logic (text, links, forms, hidden text, errors) is exercised.
    They verify OUR code's handling - not PyMuPDF itself.
    """

    def make_fake(self, pages=1, text="", needs_pass=False, fail_open=False, link_text="",
                  images=0, widgets=(), hidden=0, page_error=False, link=True):
        outer = self

        class Page:
            def get_links(self_):
                if page_error:
                    raise RuntimeError("bad page")
                return [{"kind": 2, "uri": "http://paypa1.com/x", "from": object()}] if link else []

            def get_textbox(self_, rect):
                return link_text

            def get_text(self_, mode="text"):
                if mode == "dict":
                    return {"blocks": [{"lines": [{"spans": [{"text": "x" * hidden, "color": 0xFFFFFF, "size": 10}]}]}]} if hidden else {"blocks": []}
                return text

            def get_images(self_, full=False):
                return [(1,)] * images

            def widgets(self_):
                return [mock.Mock(field_name=n) for n in widgets]

            def annots(self_):
                return []

        class Doc:
            is_encrypted = needs_pass
            is_repaired = False
            page_count = pages
            metadata = {}

            def __init__(self_):
                self_.needs_pass = needs_pass

            def embfile_count(self_):
                return 0

            def embfile_names(self_):
                return []

            def load_page(self_, n):
                return Page()

            def xref_length(self_):
                return 3

            def xref_object(self_, x, compressed=False):
                return "<< /Type /Catalog >>"

            def close(self_):
                pass

        class Tools:
            @staticmethod
            def mupdf_display_errors(v):
                pass

        class Fake:
            LINK_URI, LINK_LAUNCH, LINK_GOTOR = 2, 3, 5
            TOOLS = Tools

            @staticmethod
            def open(path):
                if fail_open:
                    raise RuntimeError("cannot open")
                return Doc()

        return Fake

    def run_with(self, fake, data=None):
        with mock.patch.object(pdf_analyzer, "fitz", fake), \
                mock.patch.object(pdf_analyzer, "PYMUPDF_AVAILABLE", True):
            return pdf(data or F.text_pdf(F.NORMAL_TEXT))

    def test_parser_failure_is_inconclusive_never_safe(self):
        r = self.run_with(self.make_fake(fail_open=True))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIn("parser failed", reasons_text(r))

    def test_password_required(self):
        r = self.run_with(self.make_fake(needs_pass=True))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIn("password", reasons_text(r))

    def test_image_only_with_pymupdf(self):
        r = self.run_with(self.make_fake(text="", images=1, link=False), F.image_only_pdf())
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIn("ocr is not available", reasons_text(r))

    def test_text_links_forms_and_mismatch_are_used(self):
        fake = self.make_fake(text="Please verify your account password immediately.",
                              link_text="https://www.paypal.com/security", widgets=("Password",))
        r = self.run_with(fake)
        text = reasons_text(r)
        self.assertEqual(r["classification"], "POTENTIALLY MALICIOUS")
        self.assertIn("misleading link", text)
        self.assertIn("sensitive information", text)
        self.assertIn("look-alike", text)

    def test_hidden_text(self):
        r = self.run_with(self.make_fake(text="A normal looking paragraph of text here.", hidden=50))
        self.assertIn("hidden text", reasons_text(r))

    def test_page_errors_block_safe(self):
        r = self.run_with(self.make_fake(text="A normal looking paragraph of text here.", page_error=True))
        self.assertNotEqual(r["classification"], "SAFE")
        self.assertIn("could not be read", reasons_text(r) + " ".join(r["notes"]).lower())

    def test_clean_document_via_pymupdf_is_safe(self):
        fake = self.make_fake(text="A perfectly ordinary paragraph about office supplies and schedules.",
                              link=False)
        r = self.run_with(fake)
        self.assertEqual(r["classification"], "SAFE")
        self.assertEqual(r["analysis_status"], "complete")
        self.assertEqual(r["analysis_engine"], "PyMuPDF + built-in scanner")


# ===========================================================================
# API + combined logic
# ===========================================================================
class ApiAndCombined(unittest.TestCase):

    def test_api_shape_and_inconclusive_has_null_score(self):
        st, j = scan(email_text="Please review the quarterly report when you can, thank you.")
        self.assertEqual(st, 200)
        self.assertTrue(j["success"])
        for key in ("classification", "risk_score", "reasons", "notes", "checks_performed", "analysis_status"):
            self.assertIn(key, j["email"])
        self.assertEqual(j["overall"]["classification"], "INCONCLUSIVE")
        self.assertIsNone(j["overall"]["risk_score"])
        self.assertIn("breakdown", j["overall"])

    def test_inconclusive_part_never_lets_overall_be_safe(self):
        st, j = scan(email_file=upload(F.EML_NORMAL, "a.eml"), pdf_file=upload(F.image_only_pdf(), "s.pdf"))
        self.assertEqual(j["email"]["classification"], "SAFE")
        self.assertEqual(j["pdf"]["classification"], "INCONCLUSIVE")
        self.assertEqual(j["overall"]["classification"], "INCONCLUSIVE")
        self.assertTrue(any("inconclusive" in r.lower() for r in j["overall"]["reasons"]))

    def test_safe_plus_safe_is_safe(self):
        st, j = scan(email_file=upload(F.EML_NORMAL, "a.eml"), pdf_file=upload(F.text_pdf(F.NORMAL_TEXT), "n.pdf"))
        self.assertEqual(j["overall"]["classification"], "SAFE")
        self.assertEqual(j["overall"]["risk_score"], 0)

    def test_inconclusive_does_not_mask_a_real_finding(self):
        st, j = scan(email_file=upload(F.EML_PHISHING_PASSWORD, "p.eml"), pdf_file=upload(F.image_only_pdf(), "s.pdf"))
        self.assertEqual(j["overall"]["classification"], "PHISHING")
        st, j = scan(email_file=upload(F.EML_NORMAL, "a.eml"), pdf_file=upload(F.text_pdf(F.PHISHING_TEXT), "p.pdf"))
        self.assertNotEqual(j["overall"]["classification"], "SAFE")

    def test_suspicious_email_floor(self):
        out = combine_results({"classification": "SUSPICIOUS", "risk_score": 30, "reasons": []},
                              {"classification": "SAFE", "risk_score": 0, "reasons": []})
        self.assertEqual(out["classification"], "SUSPICIOUS")
        self.assertGreaterEqual(out["risk_score"], 35)

    def test_both_inconclusive(self):
        out = combine_results({"classification": "INCONCLUSIVE", "risk_score": None, "reasons": ["x"]},
                              {"classification": "INCONCLUSIVE", "risk_score": None, "reasons": ["y"]})
        self.assertEqual(out["classification"], "INCONCLUSIVE")
        self.assertIsNone(out["risk_score"])

    def test_error_responses_never_contain_a_verdict(self):
        for fields in ({}, {"pdf_file": upload(b"MZ not a pdf", "e.pdf")},
                       {"pdf_file": upload(F.text_pdf(F.NORMAL_TEXT), "n.txt")},
                       {"email_file": upload("hello", "m.txt")}):
            st, j = scan(**fields)
            self.assertGreaterEqual(st, 400)
            self.assertFalse(j["success"])
            self.assertNotIn("overall", j)

    def test_analyzer_crash_is_a_500_not_safe_and_cleans_up(self):
        def crash(path):
            raise RuntimeError("secret /server/path")
        with mock.patch.object(R, "analyze_pdf", crash):
            st, j = scan(pdf_file=upload(F.text_pdf(F.NORMAL_TEXT), "n.pdf"))
        self.assertEqual(st, 500)
        self.assertNotIn("secret", json.dumps(j))
        self.assertNotIn("overall", j)
        self.assertEqual(os.listdir(R.PDF_TEMP_DIR), [])

    def test_temp_files_are_always_removed(self):
        for data in (F.text_pdf(F.NORMAL_TEXT), F.encrypted_pdf(F.NORMAL_TEXT), b"%PDF-1.4\nzzz", F.image_only_pdf()):
            scan(pdf_file=upload(data, "t.pdf"))
        self.assertEqual(os.listdir(R.PDF_TEMP_DIR), [])

    def test_existing_website_endpoints_untouched(self):
        self.assertEqual(client.get("/health").get_json()["status"], "online")
        self.assertTrue(client.get("/history").get_json()["success"])
        page = client.get("/").data
        self.assertIn(b'id="urlInput"', page)
        self.assertIn(b'id="emailPdfScanner"', page)


# ===========================================================================
# Frontend rendering (runs the real static/email_pdf.js in Node with a DOM stub)
# ===========================================================================
@unittest.skipUnless(shutil.which("node"), "Node.js is not installed - frontend rendering test skipped")
class FrontendRendering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        scenarios = []

        def add(name, status, body):
            scenarios.append({"name": name, "status": status, "body": body})

        for name, fields in {
            "safe_email": {"email_file": upload(F.EML_NORMAL, "a.eml")},
            "phishing_email": {"email_file": upload(F.EML_PHISHING_PASSWORD, "a.eml")},
            "inconclusive_email": {"email_text": "Please review the quarterly report when you can, thank you."},
            "inconclusive_pdf": {"pdf_file": upload(F.image_only_pdf(), "s.pdf")},
            "malicious_pdf": {"pdf_file": upload(F.text_pdf(F.PHISHING_TEXT), "p.pdf")},
            "safe_pdf": {"pdf_file": upload(F.text_pdf(F.NORMAL_TEXT), "n.pdf")},
            "both_mixed": {"email_file": upload(F.EML_NORMAL, "a.eml"), "pdf_file": upload(F.image_only_pdf(), "s.pdf")},
        }.items():
            st, j = scan(**fields)
            add(name, st, j)
        st, j = scan()
        add("input_error", st, j)
        ok = scan(email_file=upload(F.EML_NORMAL, "a.eml"))[1]
        broken = json.loads(json.dumps(ok))
        del broken["overall"]["classification"]
        add("missing_overall_classification", 200, broken)
        broken = json.loads(json.dumps(ok))
        del broken["overall"]
        add("missing_overall", 200, broken)
        broken = json.loads(json.dumps(ok))
        broken["email"]["classification"] = "UNKNOWN_LABEL"
        add("unknown_label", 200, broken)
        broken = json.loads(json.dumps(ok))
        del broken["email"]["reasons"]
        add("missing_reasons", 200, broken)
        add("server_error_json", 500, {"success": False, "error": "The scan could not be completed."})
        add("non_json_error", 502, None)
        add("empty_object", 200, {})

        harness = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_email_pdf_frontend_harness.js")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(scenarios, fh)
            payload = fh.name
        try:
            proc = subprocess.run(["node", harness, payload], capture_output=True, text=True, timeout=60)
        finally:
            os.remove(payload)
        if proc.returncode != 0:
            raise RuntimeError("frontend harness failed: " + proc.stderr[-800:])
        cls.out = {row["name"]: row for row in json.loads(proc.stdout)}

    def test_safe_results_render_safe(self):
        self.assertIn("SAFE", self.out["safe_email"]["overall"])
        self.assertIn("ep-safe", self.out["safe_email"]["overallClass"])
        self.assertIn("SAFE", self.out["safe_pdf"]["overall"])

    def test_phishing_and_malicious_render_as_danger(self):
        self.assertIn("PHISHING", self.out["phishing_email"]["overall"])
        self.assertIn("ep-phishing", self.out["phishing_email"]["overallClass"])
        self.assertIn("ep-malicious", self.out["malicious_pdf"]["pdfCardClass"])
        self.assertIn("POTENTIALLY MALICIOUS", self.out["malicious_pdf"]["pdfBadge"])

    def test_inconclusive_renders_as_inconclusive_with_na_score(self):
        for name in ("inconclusive_email", "inconclusive_pdf", "both_mixed"):
            row = self.out[name]
            self.assertIn("INCONCLUSIVE", row["overall"], name)
            self.assertIn("ep-inconclusive", row["overallClass"], name)
            self.assertIn("N/A", row["overallScore"], name)
            self.assertNotIn("SAFE", row["overall"], name)
            self.assertNotIn("0 / 100", row["overallScore"], name)
        self.assertIn("N/A", self.out["inconclusive_pdf"]["pdfBadge"])
        self.assertIn("OCR", self.out["inconclusive_pdf"]["pdfText"])

    def test_reasons_checks_and_notes_are_displayed(self):
        self.assertIn("CHECKS PERFORMED", self.out["safe_email"]["emailText"])
        self.assertIn("password", self.out["phishing_email"]["emailText"].lower())
        self.assertIn("wire transfer", self.out["malicious_pdf"]["pdfText"].lower())
        self.assertIn("rule-based evidence score", self.out["safe_pdf"]["pdfText"])

    def test_errors_and_unexpected_responses_never_render_a_verdict(self):
        for name in ("input_error", "missing_overall_classification", "missing_overall", "unknown_label",
                     "missing_reasons", "server_error_json", "non_json_error", "empty_object"):
            row = self.out[name]
            self.assertFalse(row["resultsVisible"], name)
            self.assertEqual(row["overall"], "", name)
            self.assertTrue(row["error"], name)
            self.assertNotIn("SAFE", row["error"].upper().replace("UNSAFE", ""), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
