"""
make_sample_files.py
====================
Creates harmless sample files in ./sample_files/ so you can try the
"Email + PDF Security Scanner" in the browser.

The "suspicious" PDFs are NOT real malware. They only contain the PDF
KEYWORDS (/JavaScript, /OpenAction, /Launch ...) that the static analyzer
looks for. They contain no working exploit. They are generated here (instead
of shipped as ready-made files) so antivirus software does not complain about
the project zip.

Run:   py -3.11 make_sample_files.py
"""

import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


def build_pdf(objects):
    """
    Build a small, structurally valid PDF from a list of object bodies (bytes).
    Object 1 must be the Catalog. Offsets in the xref table are computed.
    """
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    return bytes(out)


def _page_objects(catalog_extra=b"", extra_objects=()):
    """Catalog(1), Pages(2), Page(3), Content(4) + extras starting at object 5."""
    text = b"BT /F1 18 Tf 50 150 Td (Sample document) Tj ET"
    return [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog_extra + b" >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>",
        b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n" + text + b"\nendstream",
        *extra_objects,
    ]


def normal_pdf():
    return build_pdf(_page_objects())


def javascript_only_pdf():
    """Contains a JavaScript keyword only (like a form with calculations)."""
    return build_pdf(_page_objects(
        b"/Names << /JavaScript 5 0 R >>",
        [b"<< /Names [(calc) 6 0 R] >>", b"<< /S /JavaScript /JS (var a = 1;) >>"],
    ))


def malicious_indicators_pdf():
    """Auto-run + JavaScript + Launch KEYWORDS. Inert - for testing the detector only."""
    return build_pdf(_page_objects(
        b"/OpenAction 5 0 R",
        [b"<< /S /JavaScript /JS (var a = 1;) >>", b"<< /S /Launch /F (calc.exe) >>"],
    ))


def embedded_file_pdf():
    return build_pdf(_page_objects(
        b"/Names << /EmbeddedFiles 5 0 R >>",
        [b"<< /Names [(update.exe) 6 0 R] >>",
         b"<< /Type /Filespec /F (update.exe) /EF << /F 7 0 R >> >>",
         b"<< /Type /EmbeddedFile /Length 0 >>\nstream\n\nendstream"],
    ))


EMAILS = {
    "safe_email.eml": """From: Priya Sharma <priya@company-example.org>
To: team@company-example.org
Subject: Meeting notes from Tuesday
Content-Type: text/plain; charset="utf-8"

Hi team,

The notes from Tuesday's planning meeting are in the shared folder.
The next sync is Thursday at 3pm. Docs: https://docs.company-example.org/notes

Thanks,
Priya
""",
    "spam_email.eml": """From: Deals <deals@bulkmail-example.net>
To: you@example.com
Subject: CONGRATULATIONS WINNER!!! You have won a FREE prize
Content-Type: text/plain; charset="utf-8"

You are the lucky winner of our lottery! Claim your free gift now, 100% free,
no obligation, limited offer!!! Earn money fast, work from home, buy now!
Click here to claim $1,000,000. Unsubscribe anytime.
""",
    "phishing_email.eml": """From: "PayPal Security" <service@paypa1-secure-login.xyz>
To: victim@example.com
Reply-To: attacker@gmail.com
Subject: URGENT: Your account has been suspended - verify immediately
Authentication-Results: mx.example.com; spf=fail smtp.mailfrom=paypa1-secure-login.xyz
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html><body><p>Dear customer, we detected unusual activity. Your account will be
suspended within 24 hours. Please <a href="http://203.0.113.9/paypal/login">https://www.paypal.com/security</a>
to confirm your identity and verify your password.</p></body></html>
""",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = {
        "normal.pdf": normal_pdf(),
        "suspicious_javascript.pdf": javascript_only_pdf(),
        "suspicious_embedded_file.pdf": embedded_file_pdf(),
        "malicious_indicators_TEST_ONLY.pdf": malicious_indicators_pdf(),
    }
    for name, data in files.items():
        with open(os.path.join(OUT_DIR, name), "wb") as fh:
            fh.write(data)
    for name, text in EMAILS.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print("Sample files written to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print("  ", name)


if __name__ == "__main__":
    main()
