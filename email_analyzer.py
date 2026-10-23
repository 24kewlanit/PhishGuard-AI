"""
email_analyzer.py
=================
STAGE 2 - EMAIL PROCESSING

Turns a raw email (an uploaded .eml file OR pasted text) into a plain Python
dictionary that the rest of the project can use.

SAFETY NOTES (this is a defensive tool):
  * We only PARSE text with Python's standard `email` library.
  * HTML is converted to readable text by a small parser. <script>/<style>
    content is thrown away. Nothing is rendered or executed.
  * URLs are only extracted as strings. They are NEVER opened or followed.
  * Attachments are only described (name / type / size). They are never saved
    or opened here.

Public functions
----------------
    parse_eml_bytes(raw_bytes)  -> dict    (for an uploaded .eml file)
    parse_email_text(text)      -> dict    (for pasted content)
    build_model_text(...)       -> str     (clean text used by the ML model)
    normalize_text(text)        -> str     (shared cleaning, also used in training)
"""

import os
import re
from email import message_from_bytes, message_from_string, policy
from email.utils import parseaddr, getaddresses
from html.parser import HTMLParser
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Limits (protect the server from huge inputs)
# ---------------------------------------------------------------------------
MAX_BODY_CHARS = 200_000   # we never analyze more than this much body text
MAX_URLS = 200             # we never keep more than this many URLs

# The token that replaces every URL before ML classification.
URL_TOKEN = "urltoken"

# Matches http://..., https://... and www.... up to the next space / bracket.
URL_REGEX = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"'\)\]]+)""")

# Removes control characters (keeps normal whitespace, which is normalised later)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Domains written WITHOUT http:// or www. (e.g. "paypa1-secure.com") are still
# links a victim may type or click. Only well-known / commonly abused TLDs are
# accepted so that ordinary text such as "file.txt" or "e.g." is not mistaken
# for a domain.
_BARE_TLDS = {
    "com", "net", "org", "info", "biz", "xyz", "top", "click", "work", "support",
    "zip", "mov", "icu", "cyou", "rest", "monster", "buzz", "gq", "tk", "ml", "cf",
    "ga", "online", "site", "shop", "store", "live", "link", "app", "cc", "co", "io",
}
_BARE_DOMAIN_RE = re.compile(r"(?i)(?<![@\w./:-])((?:[a-z0-9][a-z0-9-]{0,62}\.)+[a-z]{2,24})(?![\w@-])")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _clean_url(url):
    """Strip trailing punctuation that is usually part of the sentence."""
    return url.rstrip(".,;:!?)'\"")


def extract_urls_from_text(text):
    """Return the URLs found in plain text (as strings only - never fetched)."""
    return [_clean_url(m) for m in URL_REGEX.findall(text or "")]


def _plain_host(url):
    """Hostname of a URL string (pure string parsing, never fetched)."""
    try:
        if not re.match(r"(?i)^https?://", url):
            url = "http://" + url
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def extract_bare_domains(text, existing_urls=()):
    """Domains typed without a scheme (strings only - never fetched)."""
    known = {_plain_host(u) for u in existing_urls}
    found = []
    for m in _BARE_DOMAIN_RE.finditer(text or ""):
        domain = m.group(1).lower().rstrip(".")
        if domain.rsplit(".", 1)[-1] in _BARE_TLDS and domain not in known and domain not in found:
            found.append(domain)
    return found


def normalize_text(text):
    """
    Shared cleaning used BOTH when training and when predicting, so the model
    always sees text in the same format.

      1. lower-case
      2. replace every URL with the token 'urltoken'
      3. remove control characters
      4. collapse all whitespace into single spaces
    """
    text = str(text or "").lower()
    text = _CONTROL_CHARS.sub(" ", text)
    text = URL_REGEX.sub(f" {URL_TOKEN} ", text)
    return " ".join(text.split())


def build_model_text(sender, subject, body, urls=None):
    """
    Build the single string that the email ML pipeline receives:
        Sender + Subject + Body (+ one URL token per URL found only in HTML links)
    """
    combined = f"sender: {sender or ''} subject: {subject or ''} body: {body or ''}"
    cleaned = normalize_text(combined)

    # URLs that appear inside HTML href="..." are not in the visible text,
    # so add one token for each so the model still "sees" that links exist.
    visible_url_count = cleaned.count(URL_TOKEN)
    extra = max(0, min(len(urls or []), 20) - visible_url_count)
    if extra:
        cleaned += (" " + URL_TOKEN) * extra
    return cleaned


# ---------------------------------------------------------------------------
# HTML -> readable text (safe: never executes anything)
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    """Collect visible text and (href, anchor-text) pairs from HTML."""

    _SKIP_TAGS = {"script", "style", "head", "title"}
    _BREAK_TAGS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.links = []            # list of (href, visible anchor text)
        self._skip_depth = 0
        self._href = None
        self._anchor_text = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor_text = []
        if tag in self._BREAK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "a" and self._href is not None:
            anchor = " ".join("".join(self._anchor_text).split())
            self.links.append((self._href.strip(), anchor))
            self._href = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        self.text_parts.append(data)
        if self._href is not None:
            self._anchor_text.append(data)


def html_to_text(html_str):
    """Return (readable_text, [(href, anchor_text), ...]) for an HTML string."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_str or "")
        parser.close()
    except Exception:
        # Broken HTML should never crash the scan - use what we collected.
        pass
    text = "".join(parser.text_parts)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, parser.links


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def _header(msg, name):
    """Safely read one header as a clean string ('' if missing/broken)."""
    try:
        value = msg.get(name)
        return _CONTROL_CHARS.sub("", str(value)).strip() if value is not None else ""
    except Exception:
        return ""


def _domain_of(address):
    return address.rsplit("@", 1)[1].lower().strip(">").strip() if "@" in address else ""


def _auth_results(msg):
    """
    Read SPF / DKIM / DMARC results if the receiving mail server added them.
    Only values that are really present in the headers are returned.
    """
    found = {}
    haystack = " ".join(
        _header(msg, h) for h in ("Authentication-Results", "Received-SPF")
    ).lower()
    for mech in ("spf", "dkim", "dmarc"):
        m = re.search(rf"\b{mech}\s*=\s*([a-z]+)", haystack)
        if m:
            found[mech] = m.group(1)
    # "Received-SPF: pass (...)" has no "spf=" prefix
    if "spf" not in found:
        m = re.match(r"\s*(pass|fail|softfail|neutral|none)", _header(msg, "Received-SPF").lower())
        if m:
            found["spf"] = m.group(1)
    return found


# ---------------------------------------------------------------------------
# Bulk / marketing-platform headers (evidence of mass mailing, NOT of fraud)
# ---------------------------------------------------------------------------
_BULK_HEADER_NAMES = {
    "list-unsubscribe": "List-Unsubscribe",
    "list-unsubscribe-post": "List-Unsubscribe-Post",
    "list-id": "List-Id",
    "x-mailgun-track": "X-Mailgun-Track",
    "x-mailgun-tag": "X-Mailgun-Tag",
    "x-campaign": "X-Campaign",
    "x-campaignid": "X-CampaignID",
    "x-mailchimp-id": "X-MailChimp-ID",
    "x-mc-user": "X-MC-User",
    "x-mandrill-user": "X-Mandrill-User",
    "x-sg-eid": "X-SG-EID",
    "x-sg-id": "X-SG-ID",
    "x-csa-complaints": "X-CSA-Complaints",
    "x-mailjet-campaign": "X-Mailjet-Campaign",
    "x-sib-id": "X-Sib-Id",
    "x-hs-cid": "X-HS-CID",
    "x-klaviyo-id": "X-Klaviyo-ID",
}
_BULK_MAILERS = re.compile(
    r"(?i)(mailchimp|sendgrid|mailgun|sendinblue|brevo|constant\s*contact|campaign\s*monitor|"
    r"hubspot|klaviyo|mailjet|sparkpost|getresponse|aweber|marketo|salesforce marketing|"
    r"exacttarget|braze|iterable|moengage|clevertap)")


def _bulk_headers(msg):
    """
    Names of headers that show the message was sent through a bulk-mailing /
    marketing platform. Pure header reading - nothing is contacted.
    """
    found = []
    try:
        names = {str(k).lower() for k in msg.keys()}
    except Exception:
        return found
    for low, nice in _BULK_HEADER_NAMES.items():
        if low in names:
            found.append(nice)
    if re.search(r"(?i)\b(bulk|list|junk)\b", _header(msg, "Precedence")):
        found.append("Precedence: " + _header(msg, "Precedence").lower()[:20])
    if _BULK_MAILERS.search(_header(msg, "X-Mailer")):
        found.append("X-Mailer: " + _header(msg, "X-Mailer")[:40])
    return list(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------
def _extract_body_and_attachments(msg):
    """
    Walk a (possibly multipart) message and return:
        plain_text, html_text, html_links, attachments
    Preference: text/plain for the body; HTML is used when there is no plain text.
    """
    plain_parts, html_texts, html_links, attachments = [], [], [], []

    for part in msg.walk():
        if part.is_multipart():
            continue

        disposition = part.get_content_disposition()   # 'attachment' / 'inline' / None
        content_type = part.get_content_type()
        filename = part.get_filename()

        # ---- attachments: describe only, never open ----
        if disposition == "attachment" or filename:
            try:
                size = len(part.get_payload(decode=True) or b"")
            except Exception:
                size = 0
            safe_name = os.path.basename(_CONTROL_CHARS.sub("", str(filename or "unnamed")))[:120]
            attachments.append({"filename": safe_name, "content_type": content_type, "size": size})
            continue

        # ---- body text ----
        if content_type in ("text/plain", "text/html"):
            try:
                content = part.get_content()
            except Exception:
                # Unknown/broken charset: decode leniently instead of failing.
                payload = part.get_payload(decode=True) or b""
                content = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain":
                plain_parts.append(content)
            else:
                text, links = html_to_text(content)
                html_texts.append(text)
                html_links.extend(links)

    return "\n".join(plain_parts), "\n".join(html_texts), html_links, attachments


def _unique(items, limit):
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
            if len(out) >= limit:
                break
    return out


_KNOWN_HEADERS = ("From", "Subject", "Date", "Received", "Message-ID", "To", "Reply-To", "MIME-Version")


def _readable_ratio(text):
    """Share of characters that are normal readable text (letters, digits, punctuation, space)."""
    text = text or ""
    if not text:
        return 1.0
    ok = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ".,;:!?-_'\"()[]{}<>@#$%&*+=/\\|~`^")
    return ok / len(text)


def _result_from_parts(sender_raw, reply_to, recipient, subject, plain, html_text,
                       html_links, attachments, auth, has_headers, issues=None, parse_ok=True,
                       bulk_headers=None):
    """Assemble the final dictionary used by the rest of the project."""
    issues = list(issues or [])

    # Analyse BOTH the plain-text and the HTML version when they differ: an
    # attacker can show a harmless text part and a phishing HTML part.
    plain_norm = " ".join(plain.lower().split())
    html_norm = " ".join(html_text.lower().split())
    if plain.strip() and html_text.strip() and html_norm not in plain_norm and plain_norm not in html_norm:
        body = (plain + "\n" + html_text)[:MAX_BODY_CHARS]
    else:
        body = (plain if plain.strip() else html_text)[:MAX_BODY_CHARS]

    name, address = parseaddr(sender_raw)
    reply_name, reply_addr = parseaddr(reply_to)

    # URLs: visible URLs in the text + URLs hidden inside HTML links + bare domains
    href_urls = [h for h, _ in html_links if re.match(r"(?i)^(https?://|www\.)", h)]
    text_urls = extract_urls_from_text(plain) + extract_urls_from_text(html_text) + href_urls
    bare = extract_bare_domains(plain + "\n" + html_text, text_urls)
    urls = _unique(text_urls + bare, MAX_URLS)

    # Only keep anchors whose href is a web link (used for mismatch detection)
    links = [
        {"href": h, "text": t}
        for h, t in html_links
        if re.match(r"(?i)^(https?://|www\.)", h)
    ][:MAX_URLS]

    sender_display = f"{name} <{address}>" if name and address else (address or sender_raw)

    # Unreadable / binary content must never be scanned as if it were a normal message.
    if parse_ok and body.strip() and _readable_ratio(body) < 0.6:
        parse_ok = False
        issues.append("The message content is not readable text (binary or corrupted data).")

    return {
        "sender": sender_display.strip(),
        "sender_name": name.strip(),
        "sender_address": address.strip().lower(),
        "sender_domain": _domain_of(address),
        "reply_to": reply_addr.strip().lower(),
        "reply_to_domain": _domain_of(reply_addr),
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "urls": urls,
        "links": links,                 # [{"href":..., "text":...}] from HTML emails
        "attachments": attachments,     # [{"filename","content_type","size"}]
        "auth_results": auth,           # e.g. {"spf": "fail"} if present
        "bulk_headers": list(bulk_headers or []),   # e.g. ["List-Unsubscribe", "X-Mailgun-Track"]
        "has_headers": has_headers,
        "has_sender": bool(address.strip()) and "@" in address,
        "parse_ok": parse_ok,
        "parse_issues": issues,
        "model_text": build_model_text(sender_display, subject, body, urls),
    }


def _failed_result(issue):
    """Result for input that could not be parsed at all (never treated as safe)."""
    return _result_from_parts("", "", "", "", "", "", [], [], {}, False,
                              issues=[issue], parse_ok=False)


def _parse_message(msg, has_text_headers):
    """Common logic for .eml bytes and pasted text with headers."""
    issues = []
    recipients = getaddresses([_header(msg, "To")])
    recipient = ", ".join(a for _, a in recipients if a)[:300]

    plain, html_text, html_links, attachments = _extract_body_and_attachments(msg)

    # Only count headers that really exist - an .eml without a From/Subject/...
    # line has no usable header information.
    has_headers = has_text_headers and any(_header(msg, h) for h in _KNOWN_HEADERS)
    if not has_headers:
        issues.append("No email headers (From, Subject, Date ...) were found in the input.")
    elif not parseaddr(_header(msg, "From"))[1]:
        issues.append("The message has no readable From address.")

    if msg.defects:
        issues.append("The message structure is malformed (" + ", ".join(
            sorted({type(d).__name__.replace("Defect", "") for d in msg.defects})[:3]) + ").")

    return _result_from_parts(
        sender_raw=_header(msg, "From"),
        reply_to=_header(msg, "Reply-To"),
        recipient=recipient,
        subject=_header(msg, "Subject")[:300],
        plain=plain,
        html_text=html_text,
        html_links=html_links,
        attachments=attachments,
        auth=_auth_results(msg),
        has_headers=has_headers,
        issues=issues,
        bulk_headers=_bulk_headers(msg),
    )


def parse_eml_bytes(raw_bytes):
    """Parse the bytes of an uploaded .eml file. The file is never executed."""
    try:
        msg = message_from_bytes(raw_bytes, policy=policy.default)
        return _parse_message(msg, has_text_headers=True)
    except Exception:
        return _failed_result("The .eml file could not be parsed.")


_HEADER_START = re.compile(
    r"(?im)^(from|to|subject|date|received|return-path|reply-to|mime-version|content-type):"
)


def parse_email_text(text):
    """
    Parse pasted email content.

    Two cases are handled:
      * The user pasted a full raw email WITH headers (From:, Subject:, ...)
        -> parsed exactly like an .eml file.
      * The user pasted only the message body
        -> sender/subject stay empty and the text is treated as the body.
    """
    text = (text or "")[:MAX_BODY_CHARS * 2]
    stripped = text.lstrip()

    try:
        if _HEADER_START.match(stripped):
            msg = message_from_string(stripped, policy=policy.default)
            return _parse_message(msg, has_text_headers=True)

        # Body only. If it looks like HTML, convert it to text safely.
        if re.search(r"(?i)<\s*(html|body|a\s+href|div|p)\b", text):
            html_text, html_links = html_to_text(text)
            plain = ""
        else:
            html_text, html_links, plain = "", [], text

        return _result_from_parts(
            sender_raw="", reply_to="", recipient="", subject="",
            plain=plain, html_text=html_text, html_links=html_links,
            attachments=[], auth={}, has_headers=False,
            issues=["Only the message text was supplied - no sender, Reply-To or authentication headers."],
        )
    except Exception:
        return _failed_result("The pasted email could not be parsed.")
