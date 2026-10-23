"""
email_pdf_fixtures.py
=====================
Synthetic, HARMLESS test fixtures for the Email + PDF scanner tests.

  * Nothing here is malware. The "suspicious" PDFs only contain the PDF
    KEYWORDS (/JavaScript, /OpenAction ...) or text/links that the static
    analyzer looks for. No working exploit, no real script, no real attachment.
  * Every URL uses reserved/example domains (example.com, *.example, TEST-NET
    203.0.113.x) or is a made-up look-alike string. The scanner never visits
    them and neither do the tests.
  * Fixtures are generated from CONTENT. The tests never choose a result from
    a file name.
"""

import base64
import hashlib
import struct
import zlib


# ===========================================================================
# PDF building blocks
# ===========================================================================
def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")


def assemble_pdf(objects, trailer_extra=b"", header=b"%PDF-1.4\n"):
    """Build a structurally valid PDF. objects[0] must be the Catalog."""
    out = bytearray(header)
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R ".encode() + trailer_extra +
            f">>\nstartxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(out)


def stream_object(data, filter_spec=None):
    dict_part = b"<< /Length " + str(len(data)).encode()
    if filter_spec:
        dict_part += b" /Filter " + filter_spec
    return dict_part + b" >>\nstream\n" + data + b"\nendstream"


def encode_stream(content, compress=None):
    """compress: None | 'flate' | 'a85flate'  ->  (bytes, /Filter spec)"""
    if compress == "flate":
        return zlib.compress(content), b"/FlateDecode"
    if compress == "a85flate":
        return base64.a85encode(zlib.compress(content), adobe=True), b"[/ASCII85Decode /FlateDecode]"
    return content, None


def content_for(lines):
    ops = b"BT /F1 12 Tf 14 TL 50 740 Td "
    for line in lines:
        ops += b"(" + pdf_escape(line) + b") Tj T* "
    return ops + b"ET"


FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def text_pdf(lines, compress=None, links=(), catalog_extra=b"", extra_objects=(),
             page_extra=b"", trailer_extra=b""):
    """
    A one-page PDF with the given text lines.
      links         : URLs added as /Link annotations (objects 5..)
      extra_objects : more object bodies; their numbers start at 5 + len(links)
      catalog_extra : extra Catalog entries (e.g. b"/OpenAction 6 0 R")
    """
    data, flt = encode_stream(content_for(lines), compress)
    annot_refs = b" ".join(f"{5 + i} 0 R".encode() for i in range(len(links)))
    page = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 " + FONT + b" >> >> " +
            (b"/Annots [" + annot_refs + b"] " if links else b"") + page_extra + b" >>")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog_extra + b" >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        stream_object(data, flt),
    ]
    for url in links:
        objects.append(b"<< /Type /Annot /Subtype /Link /Rect [50 700 300 720] /Border [0 0 0] "
                       b"/A << /S /URI /URI (" + pdf_escape(url) + b") >> >>")
    objects += list(extra_objects)
    return assemble_pdf(objects, trailer_extra)


def image_only_pdf():
    """A page that only draws an image (like a scan) - there is no text at all."""
    content = b"q 300 0 0 300 50 400 cm /Im1 Do Q"
    pixels = bytes([0, 255, 255, 0])
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /XObject << /Im1 5 0 R >> >> >>",
        stream_object(content),
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length 4 >>\nstream\n" + pixels + b"\nendstream",
    ]
    return assemble_pdf(objects)


def hex_glyph_text_pdf():
    """Text drawn with 2-byte glyph ids (as subset fonts do) - not decodable without a CMap."""
    glyphs = "".join(f"{g:04X}" for g in range(3, 60))
    content = b"BT /F1 12 Tf 50 740 Td <" + glyphs.encode() + b"> Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 " + FONT + b" >> >> >>",
        stream_object(content),
    ]
    return assemble_pdf(objects)


def corrupt_flate_pdf():
    """The content stream claims to be Flate-compressed but the data is damaged."""
    good = zlib.compress(content_for(["A perfectly ordinary sentence that will never be readable."]))
    damaged = good[:6] + bytes(b ^ 0xFF for b in good[6:20]) + good[20:]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 " + FONT + b" >> >> >>",
        stream_object(damaged, b"/FlateDecode"),
    ]
    return assemble_pdf(objects)


def objstm_javascript_pdf():
    """
    The /JavaScript + /OpenAction keywords are hidden INSIDE a compressed
    object stream, where a simple keyword grep cannot see them. Inert test data.
    """
    inner = b"<< /S /JavaScript /JS (var testOnly = 1;) >>"
    header = b"6 0 "
    body = header + inner
    objstm = stream_object(zlib.compress(body), b"/FlateDecode").replace(
        b"<< /Length", b"<< /Type /ObjStm /N 1 /First " + str(len(header)).encode() + b" /Length", 1)
    return text_pdf(["Quarterly summary of office supplies and meeting rooms for the team."],
                    catalog_extra=b"/OpenAction 6 0 R", extra_objects=[objstm])


def obfuscated_js_pdf():
    """/J#61vaScript = /JavaScript written with a hex escape (a scanner-evasion trick)."""
    return text_pdf(["Quarterly summary of office supplies and meeting rooms for the team."],
                    catalog_extra=b"/OpenAction 5 0 R",
                    extra_objects=[b"<< /S /J#61vaScript /JS (var testOnly = 1;) >>"])


def widget_form_pdf(field_names, submit_url=None):
    """A PDF form (AcroForm) with text fields named `field_names`."""
    base = 5
    widgets = []
    for i, name in enumerate(field_names):
        widgets.append(b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (" + pdf_escape(name) +
                       b") /Rect [50 " + str(600 - 40 * i).encode() + b" 300 " +
                       str(620 - 40 * i).encode() + b"] /F 4 >>")
    refs = b" ".join(f"{base + i} 0 R".encode() for i in range(len(widgets)))
    data = content_for(["Please complete the form below."])
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [" + refs + b"] >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 " + FONT + b" >> >> /Annots [" + refs + b"] >>",
        stream_object(data),
    ] + widgets
    return assemble_pdf(objects)


# ---------------------------------------------------------------------------
# Real (RC4 40-bit, standard security handler R2) password-protected PDF
# ---------------------------------------------------------------------------
_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _rc4(key, data):
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out, i, j = bytearray(), 0, 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(byte ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def encrypted_pdf(lines, user_password=b"secret", owner_password=b"owner-secret"):
    """A genuinely encrypted PDF: opening it needs the user password."""
    owner_key = hashlib.md5((owner_password + _PAD)[:32]).digest()[:5]
    o_entry = _rc4(owner_key, (user_password + _PAD)[:32])
    perms = struct.pack("<i", -4)
    file_id = b"0123456789abcdef"
    key = hashlib.md5((user_password + _PAD)[:32] + o_entry + perms + file_id).digest()[:5]
    u_entry = _rc4(key, _PAD)

    def object_key(num):
        return hashlib.md5(key + struct.pack("<I", num)[:3] + b"\x00\x00").digest()[:10]

    content = _rc4(object_key(4), content_for(lines))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 " + FONT + b" >> >> >>",
        stream_object(content),
        b"<< /Filter /Standard /V 1 /R 2 /O <" + o_entry.hex().encode() + b"> /U <" +
        u_entry.hex().encode() + b"> /P -4 >>",
    ]
    trailer = b"/Encrypt 5 0 R /ID [<" + file_id.hex().encode() + b"> <" + file_id.hex().encode() + b">] "
    return assemble_pdf(objects, trailer)


# ===========================================================================
# Reusable PDF texts
# ===========================================================================
NORMAL_TEXT = [
    "Office Supplies Policy - Information for staff",
    "This document explains how teams order stationery and printer paper.",
    "Orders are placed through the facilities desk on the first Monday of each month.",
    "Questions about the schedule can be sent to the facilities team by internal mail.",
    "Thank you for helping us keep the office well stocked.",
]

PHISHING_TEXT = [
    "URGENT: Your account has been suspended",
    "Verify your account password immediately to avoid permanent closure.",
    "Confirm your identity and enter your one-time password (OTP).",
    "Send the outstanding payment by wire transfer to the new bank account today.",
]

MIXED_TEXT = NORMAL_TEXT + [
    "To confirm your identity please verify your account password at paypa1-support.com/login",
]


# ===========================================================================
# EMAIL fixtures (raw .eml text)
# ===========================================================================
EML_NORMAL = """From: Priya Sharma <priya@acme-corp.example>
To: team@acme-corp.example
Subject: Meeting notes from Tuesday
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <abc123@acme-corp.example>
Authentication-Results: mx.acme-corp.example; spf=pass; dkim=pass; dmarc=pass
Content-Type: text/plain; charset="utf-8"

Hi team,

The notes from Tuesday's planning meeting are in the shared folder.
The next sync is Thursday at 3pm in the large meeting room.

Thanks,
Priya
"""

EML_PHISHING_PASSWORD = """From: IT Security <security@acme-corp-it.example>
To: staff@acme-corp.example
Subject: URGENT: verify your password
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x1@acme-corp-it.example>
Content-Type: text/plain; charset="utf-8"

Dear user,

Your password will expire within 24 hours. You must verify your account and
confirm your password immediately or your mailbox will be suspended.

Sign in here: http://acme-corp-login.example-verify.top/login
"""

EML_MISLEADING_LINK = """From: Notifications <noreply@service-mail.example>
To: you@example.org
Subject: Your statement is ready
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x2@service-mail.example>
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html><body><p>Your statement is ready.
<a href="http://secure-statement.example-host.top/view">https://www.paypal.com/statement</a></p></body></html>
"""

EML_REPLYTO_MISMATCH = """From: Accounts <accounts@vendor-co.example>
Reply-To: payments.team.9921@gmail.com
To: you@example.org
Subject: Updated remittance details
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x3@vendor-co.example>
Content-Type: text/plain; charset="utf-8"

Hello, please see our attached invoice and reply to confirm receipt.
Kind regards, Accounts
"""

EML_IMPERSONATION = """From: "Microsoft Account Team" <no-reply@micros0ft-support.example>
To: you@example.org
Subject: Unusual sign-in activity
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x4@micros0ft-support.example>
Content-Type: text/plain; charset="utf-8"

We detected unusual sign-in activity on your Microsoft account.
Please review the activity and secure your account.
"""

EML_NO_FROM = """Subject: Quarterly report
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Please review the quarterly report when you have a moment and send me your comments.
"""

EML_NO_AUTH_HEADERS = """From: Priya Sharma <priya@acme-corp.example>
To: team@acme-corp.example
Subject: Lunch on Friday
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Hi all, shall we book the usual place for lunch on Friday? Let me know how many of you are coming.
"""

EML_MIXED = """From: Amazon Orders <auto-confirm@amazon.com>
To: you@example.org
Subject: Your order has shipped
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x5@amazon.com>
Authentication-Results: mx.example.org; spf=pass; dkim=pass; dmarc=pass
Content-Type: text/plain; charset="utf-8"

Your order has shipped. Track it at https://www.amazon.com/progress-tracker
You may sign in to your account to review the order, or use this short link
to rate the delivery: https://bit.ly/rate-delivery
"""

EML_ATTACHMENT_EXE = """From: Accounts <accounts@vendor-co.example>
To: you@example.org
Subject: Invoice
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x6@vendor-co.example>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="BOUNDARY"

--BOUNDARY
Content-Type: text/plain; charset="utf-8"

Please open the attached invoice as soon as possible.
--BOUNDARY
Content-Type: application/octet-stream; name="invoice.pdf.exe"
Content-Disposition: attachment; filename="invoice.pdf.exe"
Content-Transfer-Encoding: base64

VEVTVC1EQVRBLU9OTFk=
--BOUNDARY--
"""

EML_ATTACHMENT_ONLY = """From: Accounts <accounts@vendor-co.example>
To: you@example.org
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x7@vendor-co.example>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="B2"

--B2
Content-Type: application/octet-stream; name="update.scr"
Content-Disposition: attachment; filename="update.scr"
Content-Transfer-Encoding: base64

VEVTVC1EQVRBLU9OTFk=
--B2--
"""

EML_TEXT_REFERENCE_ONLY = """From: Helpdesk <helpdesk@vendor-co.example>
To: you@example.org
Subject: Action needed
Date: Tue, 06 Oct 2026 10:00:00 +0000
Message-ID: <x8@vendor-co.example>
Content-Type: text/plain; charset="utf-8"

Please download and run security-update.exe from the shared drive to finish the upgrade.
"""

# --- a realistic legitimate corpus (all with a sender, used to measure false positives) ---
LEGIT_EMAILS = [
    EML_NORMAL,
    EML_NO_AUTH_HEADERS,
    """From: HR Team <hr@acme-corp.example>
To: all@acme-corp.example
Subject: Holiday calendar for next year
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Hello everyone, the holiday calendar for next year is attached to the intranet page.
Please plan your leave early and tell your manager about long breaks.
""",
    """From: Ravi Kumar <ravi.kumar@acme-corp.example>
To: team@acme-corp.example
Subject: Re: Sprint review
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Thanks Anita, I have updated the board. The review is at 4pm tomorrow. See docs at https://docs.acme-corp.example/sprint
""",
    """From: GitHub <noreply@github.com>
To: dev@example.org
Subject: [repo] Pull request merged
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Your pull request was merged into main. View it at https://github.com/example/repo/pull/42
""",
    """From: Library <desk@citylibrary.example>
To: member@example.org
Subject: Your book is ready for pickup
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

The book you reserved is ready. Please collect it from the front desk within seven days.
""",
    """From: Maya <maya@friends.example>
To: you@example.org
Subject: Weekend plans
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Are we still going hiking on Saturday? I can bring sandwiches and a map of the trail.
""",
    """From: Google <no-reply@accounts.google.com>
To: you@example.org
Subject: Security alert
Date: Tue, 06 Oct 2026 10:00:00 +0000
Authentication-Results: mx.example.org; spf=pass; dkim=pass; dmarc=pass
Content-Type: text/plain; charset="utf-8"

A new sign-in was detected. If this was you, no action is needed. Review activity at https://myaccount.google.com/notifications
""",
    """From: Accounts <billing@cloudhost.example>
To: you@example.org
Subject: Invoice 10442 for September
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Your invoice for September is available in the customer portal at https://portal.cloudhost.example/invoices
Payment is due by the 15th. Thank you for your business.
""",
    """From: Coach <coach@sportsclub.example>
To: parents@example.org
Subject: Practice moved to 6pm
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Practice has moved to 6pm on Thursday because the field is being repaired. Please bring water bottles.
""",
    """From: Conference <info@devconf.example>
To: you@example.org
Subject: Your ticket confirmation
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Thanks for registering. Your ticket is attached. Doors open at 9am. Schedule: https://devconf.example/schedule
""",
]

# --- a phishing / fraud corpus (different wording / structure; none may come out SAFE) ---
PHISHING_EMAILS = [
    EML_PHISHING_PASSWORD,
    EML_IMPERSONATION,
    EML_MISLEADING_LINK,
    EML_ATTACHMENT_EXE,
    EML_ATTACHMENT_ONLY,
    EML_TEXT_REFERENCE_ONLY,
    EML_REPLYTO_MISMATCH,
    """From: "PayPal Service" <service@paypa1-secure.example>
To: you@example.org
Subject: Action required
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Dear customer, we noticed suspicious activity. Confirm your identity and update your billing information at paypa1-secure.com now.
""",
    """From: CEO <ceo.office.mail@gmail.com>
To: finance@acme-corp.example
Subject: Quick favour
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

I am in a meeting and cannot talk. Please buy gift cards for a client and send me the codes.
Keep this confidential and do not tell anyone. I will reimburse you.
""",
    """From: Bank Alerts <alerts@hdfc-netbanking-verify.example>
To: you@example.org
Subject: Your account will be blocked
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Your KYC is incomplete. Your account will be suspended today. Update your account number and OTP here: http://203.0.113.50/hdfc/kyc
""",
    """From: Support <support@dropb0x-files.example>
To: you@example.org
Subject: A file was shared with you
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Someone shared a document. Sign in to view it: http://dropb0x-files.example/login
""",
    """From: Docs <docs@acme-corp.example>
Reply-To: attacker@evil-reply.example
To: you@example.org
Subject: Statement
Date: Tue, 06 Oct 2026 10:00:00 +0000
Authentication-Results: mx.example.org; spf=fail; dkim=fail; dmarc=fail
Content-Type: text/plain; charset="utf-8"

Please review the statement and reply with the requested details.
""",
    """From: Prize Desk <prize@lottery-win.example>
To: you@example.org
Subject: Claim your reward
Date: Tue, 06 Oct 2026 10:00:00 +0000
Content-Type: text/plain; charset="utf-8"

Download the attached form, enable macros, and pay the processing fee by Western Union to receive your prize.
""",
]
