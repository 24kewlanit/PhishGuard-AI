"""
pdf_analyzer.py
===============
STAGE 4 - PDF PROCESSING (static analysis only)

IMPORTANT - what "static analysis" means here
---------------------------------------------
The PDF is PARSED for static inspection; it is NOT executed.
  * We never open it in a browser or PDF viewer, and never launch it with the OS.
  * We never render pages to images.
  * JavaScript inside the PDF is only DETECTED (its keyword is counted); it is
    never run. Links inside the PDF are only listed; they are never visited.
  * Nothing is sent to any external website.

(To be precise: PyMuPDF has to parse the PDF's internal structure to read it.
Parsing is not execution, but like any parser it is software that processes
untrusted input - so keep PyMuPDF up to date.)

How the analysis works - two complementary layers
-------------------------------------------------
  1. PyMuPDF (fitz)   : page count, metadata, encryption, embedded files,
                        link annotations, and a walk over every PDF object
                        (this also sees objects hidden inside compressed
                        "object streams").
  2. Raw keyword scan : reads the file as bytes (after removing the binary
                        stream data) and counts PDF names such as /JavaScript,
                        /OpenAction, /Launch. It also decodes hex-obfuscated
                        names like /J#61vaScript that try to dodge scanners.

If PyMuPDF is not installed the analyzer still runs the built-in scanner (a
dependency-free parser that inflates streams and compressed object streams,
extracts text, and reads URIs / annotations / form fields), but says so in the
"notes" field.

ACCURACY RULES (the "Safe" bug fix)
-----------------------------------
  * The visible text is extracted and checked for phishing wording, fake
    payment requests, look-alike domains and impersonation.
  * Every URL (link annotations, URI actions AND URLs written in the text) is
    checked with the same look-alike / shortener / IP logic as the email scanner.
  * Anything that stops the analysis from being complete (corrupt file,
    password, scanned/image-only pages, no extractable text, undecodable
    streams, parser failure, truncated scan) is recorded in
    "analysis_issues". A PDF with such an issue and no other evidence is
    reported INCONCLUSIVE - it is never reported SAFE.

Public function
---------------
    analyze_pdf(path) -> dict          (facts + classification)
    extract_pdf_findings(path) -> dict (facts only, used by train_pdf_model.py)
"""

import base64
import binascii
import os
import re
import zlib
from collections import Counter

from email_analyzer import extract_bare_domains, extract_urls_from_text
from email_model import link_text_mismatch
from pdf_model import DANGEROUS_EMBEDDED_EXT, classify_pdf, load_pdf_model

# PyMuPDF is imported as `pymupdf` in new versions and `fitz` in older ones.
try:
    import pymupdf as fitz
except ImportError:                      # pragma: no cover
    try:
        import fitz
    except ImportError:
        fitz = None

if fitz is not None and not hasattr(fitz, "open"):   # wrong package named "fitz"
    fitz = None

PYMUPDF_AVAILABLE = fitz is not None

# ---- safety limits so a hostile file cannot make the server work forever ----
MAX_RAW_SCAN_BYTES = 20 * 1024 * 1024
MAX_PAGES_SCANNED = 300
MAX_OBJECTS_SCANNED = 100_000
MAX_OBJECT_TEXT = 20_000
MAX_URLS = 200
MAX_EMBEDDED_LISTED = 50
MAX_STREAM_DECODE = 8 * 1024 * 1024     # never inflate more than this per stream
MAX_TEXT_CHARS = 200_000
MAX_TEXT_PAGES = 100
MAX_HIDDEN_TEXT_PAGES = 30
MIN_TEXT_CHARS = 10                     # fewer characters => "no extractable text"

# PDF "names" (keywords) we count.
INTERESTING_NAMES = {
    "JavaScript", "JS", "OpenAction", "AA", "Launch", "SubmitForm", "ImportData",
    "RichMedia", "EmbeddedFile", "EmbeddedFiles", "XFA", "GoToR", "GoToE", "URI",
    "AcroForm", "JBIG2Decode", "Movie", "Sound", "Rendition", "Encrypt", "ObjStm",
    "FileAttachment",
}
# Subset shown to the user as "suspicious objects"
SUSPICIOUS_OBJECT_NAMES = [
    "JavaScript", "JS", "OpenAction", "AA", "Launch", "SubmitForm", "ImportData",
    "RichMedia", "EmbeddedFile", "XFA", "GoToR", "GoToE", "JBIG2Decode",
    "Movie", "Sound", "Rendition", "FileAttachment",
]

_NAME_RE_BYTES = re.compile(rb"/([A-Za-z0-9_#.\-]+)")
_NAME_RE_STR = re.compile(r"/([A-Za-z0-9_#.\-]+)")
_HEX_ESCAPE_BYTES = re.compile(rb"#([0-9a-fA-F]{2})")
_HEX_ESCAPE_STR = re.compile(r"#([0-9a-fA-F]{2})")
_STREAM_RE = re.compile(rb"(?<!end)stream\r?\n.*?endstream", re.S)
_URI_RE = re.compile(rb"/URI\s*\(((?:[^()\\]|\\.)*)\)")
_FILESPEC_NAME_RE = re.compile(rb"/(?:UF|F)\s*\(([^()]{1,120})\)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _decode_name(raw):
    """/J#61vaScript -> 'JavaScript'  (returns decoded_name, was_obfuscated)."""
    if isinstance(raw, bytes):
        decoded = _HEX_ESCAPE_BYTES.sub(lambda m: bytes([int(m.group(1), 16)]), raw).decode("latin-1")
        return decoded, b"#" in raw
    decoded = _HEX_ESCAPE_STR.sub(lambda m: chr(int(m.group(1), 16)), raw)
    return decoded, "#" in raw


def _clean_url(u):
    u = _CONTROL.sub("", u).strip()
    return u[:300]


# ---------------------------------------------------------------------------
# Small PDF-parsing helpers (standard library only - nothing is executed)
# ---------------------------------------------------------------------------
_OBJ_RE = re.compile(rb"(?<!\d)(\d{1,7})\s+(\d{1,5})\s+obj\b(.*?)\bendobj", re.S)
_STREAM_START_RE = re.compile(rb"\bstream\r?\n")
_PAGE_RE = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
_IMAGE_RE = re.compile(rb"/Subtype\s*/Image(?![A-Za-z])")
_ANNOT_RE = re.compile(rb"/Subtype\s*/(Link|Widget|FileAttachment|FreeText|Screen|Movie|Sound|RichMedia|3D)(?![A-Za-z])")
_FIELD_NAME_RE = re.compile(rb"/T\s*\(((?:[^()\\]|\\.){1,80})\)")
_URI_HEX_RE = re.compile(rb"/URI\s*<([0-9A-Fa-f\s]+)>")
_TEXT_TOKEN_RE = re.compile(
    rb"(\((?:\\.|[^\\()])*\))|(<[0-9A-Fa-f\s]*>)|(-?\d*\.?\d+)|(\[)|(\])|(T\*|Td|TD|Tm|ET|'|\")")
_BT_RE = re.compile(rb"\bBT\b")
_IMAGE_ONLY_CODECS = {"DCTDecode", "DCT", "JPXDecode", "CCITTFaxDecode", "CCF", "JBIG2Decode"}
_ESCAPES = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12,
            ord("("): 40, ord(")"): 41, ord("\\"): 92}


def _unescape_literal(raw):
    """Decode the bytes of a PDF literal string such as  Hello\\(World\\)  ."""
    out, i, n = bytearray(), 0, len(raw)
    while i < n:
        c = raw[i]
        if c != 0x5C:                      # not a backslash
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        c = raw[i]
        if c in _ESCAPES:
            out.append(_ESCAPES[c])
            i += 1
        elif 0x30 <= c <= 0x37:            # octal escape \ddd
            j = i
            while j < n and j < i + 3 and 0x30 <= raw[j] <= 0x37:
                j += 1
            out.append(int(raw[i:j], 8) & 0xFF)
            i = j
        elif c in (10, 13):                # line continuation
            i += 1
            if c == 13 and i < n and raw[i] == 10:
                i += 1
        else:
            out.append(c)
            i += 1
    return bytes(out)


def _decode_hex_string(raw):
    """Return (text, undecodable_char_count) for a <...> hex string."""
    clean = re.sub(rb"\s+", b"", raw)
    if len(clean) % 2:
        clean += b"0"
    try:
        data = binascii.unhexlify(clean)
    except (binascii.Error, ValueError):
        return "", 0
    if len(data) >= 2 and len(data) % 2 == 0 and all(data[i] == 0 for i in range(0, len(data), 2)):
        text = data.decode("utf-16-be", errors="replace")             # 00 41 00 42 -> "AB"
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
        if printable >= 0.8 * len(text):
            return text, 0
        return "", len(data)                                          # glyph ids, not characters
    if data and all(32 <= b < 127 or b in (9, 10, 13) for b in data):
        return data.decode("latin-1"), 0                              # plain single-byte text
    return "", len(data)                                              # glyph ids: cannot be decoded here


def _inflate(data):
    d = zlib.decompressobj()
    return d.decompress(data, MAX_STREAM_DECODE)


def _decode_stream(dict_bytes, data):
    """
    Apply the stream's /Filter chain. Returns (decoded_bytes | None, problem | None).
    problem is a short code: a filter name that is not supported, or "corrupt".
    """
    m = re.search(rb"/Filter\s*(\[[^\]]*\]|/[A-Za-z0-9]+)", dict_bytes)
    names = [n.decode("latin-1") for n in re.findall(rb"/([A-Za-z0-9]+)", m.group(1))] if m else []
    for name in names:
        try:
            if name in ("FlateDecode", "Fl"):
                if re.search(rb"/Predictor\s*(1[0-5]|2)\b", dict_bytes):
                    return None, "Predictor"
                data = _inflate(data)
            elif name in ("ASCII85Decode", "A85"):
                body = data.strip()
                if body.startswith(b"<~"):
                    body = body[2:]
                if body.endswith(b"~>"):
                    body = body[:-2]
                data = base64.a85decode(body, adobe=False, ignorechars=b" \t\n\r\v\f")
            elif name in ("ASCIIHexDecode", "AHx"):
                body = re.sub(rb"\s+", b"", data).rstrip(b">")
                data = binascii.unhexlify(body + (b"0" if len(body) % 2 else b""))
            else:
                return None, name                      # LZW, RunLength, image codecs, Crypt ...
        except Exception:
            return None, "corrupt"
    return data, None


def _split_top_level_dicts(blob):
    """Yield the top-level << ... >> dictionaries in a byte blob (depth counting)."""
    depth, start = 0, None
    for m in re.finditer(rb"<<|>>", blob):
        if m.group() == b"<<":
            if depth == 0:
                start = m.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield blob[start:m.end()]
                start = None


def _extract_content_text(content):
    """
    Pull the visible text out of a decoded page content stream (BT ... ET).
    Returns (text, hex_undecodable_chars). Text is only READ, never executed.
    """
    parts, in_array, undecodable = [], False, 0
    for m in _TEXT_TOKEN_RE.finditer(content):
        lit, hexs, num, lb, rb_, op = m.groups()
        if lit is not None:
            parts.append(_unescape_literal(lit[1:-1]).decode("cp1252", errors="replace"))
        elif hexs is not None:
            text, bad = _decode_hex_string(hexs[1:-1])
            parts.append(text)
            undecodable += bad
        elif num is not None:
            if in_array and float(num) <= -180:
                parts.append(" ")
        elif lb is not None:
            in_array = True
        elif rb_ is not None:
            in_array = False
        elif op is not None:
            parts.append("\n")
    return "".join(parts), undecodable


def _uris_from(blob):
    """URI action targets found in a byte blob ((...) and <hex> forms)."""
    found = [_clean_url(_unescape_literal(m).decode("latin-1")) for m in _URI_RE.findall(blob)]
    for m in _URI_HEX_RE.findall(blob):
        try:
            found.append(_clean_url(binascii.unhexlify(re.sub(rb"\s+", b"", m)).decode("latin-1")))
        except (binascii.Error, ValueError):
            pass
    return found


# ---------------------------------------------------------------------------
# Layer 2: built-in scanner (works even without PyMuPDF)
# ---------------------------------------------------------------------------
def _raw_scan(path):
    """
    Dependency-free static scan of the file bytes:
      * counts interesting PDF names (also inside compressed object streams),
      * collects URI targets, annotation types and form-field names,
      * inflates content streams and extracts the visible text.
    """
    with open(path, "rb") as fh:
        data = fh.read(MAX_RAW_SCAN_BYTES)

    header_idx = data[:1024].find(b"%PDF-")
    header_offset = header_idx if header_idx > 0 else 0

    structure = _STREAM_RE.sub(b"", data)      # remove binary stream payloads (avoid false hits)

    counts, obfuscated = Counter(), set()
    for raw in _NAME_RE_BYTES.findall(structure):
        name, was_obf = _decode_name(raw)
        if name in INTERESTING_NAMES:
            counts[name] += 1
            if was_obf:
                obfuscated.add(name)

    urls = _uris_from(structure)
    file_names = [_CONTROL.sub("", m.decode("latin-1"))[:120] for m in _FILESPEC_NAME_RE.findall(structure)]

    info = {
        "object_count": 0, "page_objects": 0, "image_objects": 0,
        "annot_types": Counter(), "form_fields": [],
        "text_parts": [], "undecodable_hex_chars": 0,
        "undecodable_streams": 0, "unsupported_filters": set(), "failed_objstm": 0,
        "encrypt": counts["Encrypt"] > 0,
    }
    text_len = 0

    def scan_dict_text(blob):
        info["page_objects"] += len(_PAGE_RE.findall(blob))
        info["image_objects"] += len(_IMAGE_RE.findall(blob))
        for sub in _ANNOT_RE.findall(blob):
            info["annot_types"][sub.decode()] += 1
        for d in _split_top_level_dicts(blob):
            if (b"/FT" in d or b"/Widget" in d) and len(info["form_fields"]) < 200:
                for nm in _FIELD_NAME_RE.findall(d):
                    info["form_fields"].append(_unescape_literal(nm).decode("latin-1")[:80])

    for m in _OBJ_RE.finditer(data):
        info["object_count"] += 1
        body = m.group(3)
        sm = _STREAM_START_RE.search(body)
        dict_bytes = body[:sm.start()] if sm else body
        scan_dict_text(dict_bytes)
        if not sm:
            continue
        end = body.rfind(b"endstream")
        stream = body[sm.end(): end if end > sm.end() else len(body)]
        is_objstm = re.search(rb"/Type\s*/ObjStm", dict_bytes) is not None
        is_image = _IMAGE_RE.search(dict_bytes) is not None
        if is_image or re.search(rb"/Type\s*/(XRef|EmbeddedFile)", dict_bytes):
            continue
        decoded, problem = _decode_stream(dict_bytes, stream)
        if decoded is None:
            if problem in _IMAGE_ONLY_CODECS:
                continue
            if is_objstm:
                info["failed_objstm"] += 1
            if problem == "corrupt":
                info["undecodable_streams"] += 1
            else:
                info["unsupported_filters"].add(problem)
            continue
        if is_objstm:
            # names / URIs hidden inside compressed object streams
            for raw in _NAME_RE_BYTES.findall(decoded):
                name, was_obf = _decode_name(raw)
                if name in INTERESTING_NAMES:
                    counts[name] += 1
                    if was_obf:
                        obfuscated.add(name)
            urls.extend(_uris_from(decoded))
            scan_dict_text(decoded)
        elif _BT_RE.search(decoded) and text_len < MAX_TEXT_CHARS:
            text, bad = _extract_content_text(decoded)
            info["undecodable_hex_chars"] += bad
            if text.strip():
                info["text_parts"].append(text)
                text_len += len(text)

    return counts, sorted(obfuscated), urls, header_offset, file_names, info


# ---------------------------------------------------------------------------
# Layer 1: PyMuPDF structured scan
# ---------------------------------------------------------------------------
def _new_mu_result():
    return {
        "parse_ok": True, "parse_error": None, "pages": 0, "encrypted": False,
        "password_required": False, "metadata": {}, "embedded_files": 0,
        "embedded_names": [], "urls": [], "counts": Counter(), "obfuscated": set(),
        "repaired": False,
        "text": "", "image_pages": 0, "page_errors": 0, "link_mismatches": [],
        "form_fields": [], "annot_types": Counter(), "hidden_chars": 0,
        "pages_truncated": False, "objects_truncated": False,
    }


def _pymupdf_scan(path):
    """Return a dict of facts read with PyMuPDF (static parsing only)."""
    out = _new_mu_result()
    try:
        fitz.TOOLS.mupdf_display_errors(False)      # keep the server console quiet
    except Exception:
        pass

    doc = None
    try:
        doc = fitz.open(path)
        out["encrypted"] = bool(doc.is_encrypted)
        if doc.needs_pass:
            out["password_required"] = True          # cannot read further without password
            return out
        out["repaired"] = bool(getattr(doc, "is_repaired", False))

        out["pages"] = int(doc.page_count)
        out["metadata"] = {
            k: _CONTROL.sub("", str(v))[:200]
            for k, v in (doc.metadata or {}).items() if v
        }

        # ---- embedded files ----
        try:
            out["embedded_files"] = int(doc.embfile_count())
            for name in list(doc.embfile_names())[:MAX_EMBEDDED_LISTED]:
                info = doc.embfile_info(name) or {}
                out["embedded_names"].append(
                    _CONTROL.sub("", str(info.get("filename") or info.get("ufilename") or name))[:120]
                )
        except Exception:
            out["page_errors"] += 1

        # ---- per page: links, text, images, forms, annotations (no rendering) ----
        link_uri = getattr(fitz, "LINK_URI", 2)
        link_launch = getattr(fitz, "LINK_LAUNCH", 3)
        link_gotor = getattr(fitz, "LINK_GOTOR", 5)
        n_pages = min(out["pages"], MAX_PAGES_SCANNED)
        out["pages_truncated"] = out["pages"] > MAX_PAGES_SCANNED
        text_parts, text_len = [], 0
        for pno in range(n_pages):
            try:
                page = doc.load_page(pno)
            except Exception:
                out["page_errors"] += 1
                continue

            try:
                for link in page.get_links():
                    kind = link.get("kind")
                    if kind == link_uri and link.get("uri"):
                        uri = _clean_url(str(link["uri"]))
                        out["urls"].append(uri)
                        try:
                            shown = page.get_textbox(link["from"]) if link.get("from") is not None else ""
                            reason = link_text_mismatch(shown, uri)
                            if reason and reason not in out["link_mismatches"]:
                                out["link_mismatches"].append(reason)
                        except Exception:
                            pass
                    elif kind == link_launch:
                        out["counts"]["Launch"] += 1
                    elif kind == link_gotor:
                        out["counts"]["GoToR"] += 1
            except Exception:
                out["page_errors"] += 1

            if pno < MAX_TEXT_PAGES and text_len < MAX_TEXT_CHARS:
                try:
                    page_text = page.get_text("text") or ""
                    text_parts.append(page_text)
                    text_len += len(page_text)
                except Exception:
                    out["page_errors"] += 1
            try:
                if page.get_images(full=False):
                    out["image_pages"] += 1
            except Exception:
                out["page_errors"] += 1

            try:
                for widget in (page.widgets() or []):
                    if len(out["form_fields"]) < 200:
                        out["form_fields"].append(_CONTROL.sub("", str(widget.field_name or ""))[:80])
            except Exception:
                out["page_errors"] += 1
            try:
                for annot in (page.annots() or []):
                    out["annot_types"][str(annot.type[1])] += 1
            except Exception:
                out["page_errors"] += 1

            if pno < MAX_HIDDEN_TEXT_PAGES:
                try:
                    for block in page.get_text("dict").get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                chars = len((span.get("text") or "").strip())
                                if chars and (span.get("color") == 0xFFFFFF or float(span.get("size", 10)) < 2):
                                    out["hidden_chars"] += chars
                except Exception:
                    out["page_errors"] += 1
        out["text"] = "\n".join(text_parts)[:MAX_TEXT_CHARS]

        # ---- walk every PDF object (also sees compressed object streams) ----
        xref_len = int(doc.xref_length())
        out["objects_truncated"] = xref_len - 1 > MAX_OBJECTS_SCANNED
        for xref in range(1, min(xref_len, MAX_OBJECTS_SCANNED)):
            try:
                obj = doc.xref_object(xref, compressed=False)
            except Exception:
                out["page_errors"] += 1
                continue
            if not obj:
                continue
            obj = obj[:MAX_OBJECT_TEXT]
            for raw in _NAME_RE_STR.findall(obj):
                name, was_obf = _decode_name(raw)
                if name in INTERESTING_NAMES:
                    out["counts"][name] += 1
                    if was_obf:
                        out["obfuscated"].add(name)
            if "/URI" in obj:
                for m in re.findall(r"/URI\s*\(((?:[^()\\]|\\.)*)\)", obj):
                    out["urls"].append(_clean_url(m))
    except Exception as exc:
        out["parse_ok"] = False
        out["parse_error"] = type(exc).__name__      # type only - no paths / details leaked
    finally:
        if doc is not None:
            try:
                doc.close()                 # IMPORTANT on Windows: lets the temp file be deleted
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _failed_findings(file_size, issue, engine="none"):
    """Findings for a PDF that could not be analysed at all (never means 'safe')."""
    return {
        "engine": engine, "file_size": file_size, "pages": 0, "encrypted": False,
        "password_required": False, "metadata": {}, "urls": [], "embedded_files": 0,
        "embedded_names": [], "dangerous_embedded_names": [], "javascript": False,
        "javascript_count": 0, "open_action": 0, "additional_actions": 0, "launch_action": 0,
        "submit_form": 0, "import_data": 0, "rich_media": 0, "xfa": 0, "goto_remote": 0,
        "jbig2": 0, "obfuscated_names": [], "header_offset": 0, "parse_ok": False,
        "object_counts": {}, "notes": [], "text": "", "text_chars": 0,
        "text_source": None, "text_extraction": "failed", "image_only": False,
        "link_mismatches": [], "form_fields": [], "annotation_types": {},
        "hidden_chars": 0, "repaired": False, "acroform": 0, "file_attachment": 0,
        "encryption_blocks_analysis": False,
        "analysis_issues": [issue], "analysis_status": "failed", "checks_performed": [],
    }


def extract_pdf_findings(path):
    """Collect the static facts about a PDF (no classification yet)."""
    file_size = os.path.getsize(path)
    notes, issues, checks = [], [], []

    try:
        (raw_counts, raw_obf, raw_urls, header_offset,
         raw_file_names, raw_info) = _raw_scan(path)
        raw_ok = True
    except Exception:
        raw_ok = False
        raw_counts, raw_obf, raw_urls, header_offset, raw_file_names = Counter(), [], [], 0, []
        raw_info = {"object_count": 0, "page_objects": 0, "image_objects": 0, "annot_types": Counter(),
                    "form_fields": [], "text_parts": [], "undecodable_hex_chars": 0,
                    "undecodable_streams": 0, "unsupported_filters": set(), "failed_objstm": 0,
                    "encrypt": False}
        issues.append("The file could not be read by the built-in PDF scanner.")

    if PYMUPDF_AVAILABLE:
        mu = _pymupdf_scan(path)
        engine = "PyMuPDF + built-in scanner"
    else:
        mu = _new_mu_result()
        engine = "built-in scanner only"
        notes.append("PyMuPDF is not installed - install it (pip install PyMuPDF) for page-accurate text, "
                     "metadata, embedded-file, form and link inspection. The built-in scanner was used instead.")

    # Keep the HIGHER count from the layers for each keyword.
    counts = Counter()
    for name in INTERESTING_NAMES:
        counts[name] = max(raw_counts.get(name, 0), mu["counts"].get(name, 0))
    obfuscated = sorted(set(raw_obf) | set(mu["obfuscated"]))

    encrypted = mu["encrypted"] or counts["Encrypt"] > 0
    pages = mu["pages"] or raw_info["page_objects"]

    # ------------------------------------------------------------------
    # Text, images and URLs
    # ------------------------------------------------------------------
    raw_text = "\n".join(raw_info["text_parts"])[:MAX_TEXT_CHARS]
    text, text_source = "", None
    if PYMUPDF_AVAILABLE and mu["parse_ok"] and len(mu["text"].strip()) >= MIN_TEXT_CHARS:
        text, text_source = mu["text"], "PyMuPDF"
    elif len(raw_text.strip()) >= 1:
        text, text_source = raw_text, "built-in scanner"
    elif mu["text"].strip():
        text, text_source = mu["text"], "PyMuPDF"
    text_chars = len(text.strip())
    image_pages = mu["image_pages"] or (raw_info["image_objects"] if raw_info["image_objects"] else 0)
    image_only = bool(pages and text_chars < MIN_TEXT_CHARS and image_pages)

    embedded_count = max(mu["embedded_files"], counts["EmbeddedFile"])
    embedded_names = list(mu["embedded_names"])
    if embedded_count and not embedded_names:
        embedded_names = list(dict.fromkeys(raw_file_names))[:MAX_EMBEDDED_LISTED]
    dangerous = [n for n in embedded_names
                 if n.lower().rsplit(".", 1)[-1] in DANGEROUS_EMBEDDED_EXT and "." in n]

    # URLs: link annotations / URI actions + URLs written in the text (strings only)
    link_urls = list(mu["urls"]) + list(raw_urls)
    text_urls = extract_urls_from_text(text)
    text_urls += extract_bare_domains(text, text_urls)
    seen, urls = set(), []
    for u in link_urls + [_clean_url(t) for t in text_urls]:
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
            if len(urls) >= MAX_URLS:
                break

    form_fields = list(dict.fromkeys(list(mu["form_fields"]) + raw_info["form_fields"]))
    annot_types = Counter(raw_info["annot_types"])
    for k, v in mu["annot_types"].items():
        annot_types[k] = max(annot_types.get(k, 0), v)

    # ------------------------------------------------------------------
    # What stopped the analysis from being complete?  (=> never "SAFE")
    # ------------------------------------------------------------------
    encryption_blocks = False
    blocked = not raw_ok                    # True => the file could not be inspected at all
    if mu["password_required"]:
        encryption_blocks = blocked = True
        issues.append("The PDF needs a password to open, so its contents could not be inspected.")
    elif PYMUPDF_AVAILABLE and not mu["parse_ok"]:
        blocked = True
        issues.append("The PDF parser failed (" + str(mu["parse_error"]) + "): the file is corrupt, "
                      "truncated or deliberately malformed, so it could not be inspected.")
    elif raw_ok and not PYMUPDF_AVAILABLE and raw_info["object_count"] == 0:
        blocked = True
        issues.append("No PDF objects could be parsed - the file is corrupt, truncated or not a real PDF.")
    elif pages == 0 and not (raw_info["encrypt"] and not PYMUPDF_AVAILABLE):
        blocked = True
        issues.append("The PDF contains no readable pages.")

    if raw_info["encrypt"] and not PYMUPDF_AVAILABLE and not mu["password_required"]:
        encryption_blocks = blocked = True
        issues.append("The PDF is encrypted; the built-in scanner cannot decrypt it, so the contents "
                      "could not be inspected (install PyMuPDF for encrypted-PDF support).")
    elif not PYMUPDF_AVAILABLE and raw_info["undecodable_streams"] and not raw_info["encrypt"]:
        issues.append("Some compressed streams in the PDF are damaged and could not be decoded.")
    if not PYMUPDF_AVAILABLE and raw_info["unsupported_filters"]:
        issues.append("Some streams use a compression format the built-in scanner cannot read ("
                      + ", ".join(sorted(raw_info["unsupported_filters"])[:3]) + ").")
    if not PYMUPDF_AVAILABLE and raw_info["failed_objstm"]:
        issues.append("Compressed object streams could not be opened, so hidden objects may exist.")

    if mu["page_errors"]:
        issues.append(f"{mu['page_errors']} page(s)/object(s) could not be read by the parser.")
    if mu["pages_truncated"]:
        issues.append(f"Only the first {MAX_PAGES_SCANNED} pages were inspected.")
    if mu["objects_truncated"]:
        issues.append(f"Only the first {MAX_OBJECTS_SCANNED:,} PDF objects were inspected.")

    # text extraction status
    if blocked:
        text_status = "failed"
    elif pages and text_chars < MIN_TEXT_CHARS:
        text_status = "none"
        if image_only:
            issues.append("The pages contain images but no extractable text (scanned or image-only PDF). "
                          "OCR is not available, so any text inside the images could not be checked.")
        elif raw_info["undecodable_hex_chars"] and not PYMUPDF_AVAILABLE:
            issues.append("The text uses an embedded font encoding that the built-in scanner cannot decode "
                          "(install PyMuPDF for full text extraction).")
        else:
            issues.append("No extractable text was found, so the wording of the document could not be checked.")
    else:
        text_status = "ok"
        if (not PYMUPDF_AVAILABLE and raw_info["undecodable_hex_chars"]
                and raw_info["undecodable_hex_chars"] > text_chars):
            text_status = "partial"
            issues.append("Most of the text uses a font encoding the built-in scanner cannot decode, so only "
                          "part of the wording could be checked (install PyMuPDF for full extraction).")

    if mu["repaired"]:
        notes.append("The PDF structure was damaged and had to be repaired by the parser.")
    if not mu["parse_ok"] and PYMUPDF_AVAILABLE:
        notes.append("The PDF could not be fully parsed (it may be corrupt or deliberately malformed).")
    if encrypted and not encryption_blocks:
        notes.append("The PDF is encrypted but could be opened without a password and was inspected.")

    # ---- checks that really ran (shown to the user, also for SAFE results) ----
    if raw_ok and raw_info["object_count"]:
        checks.append("PDF structure: JavaScript, auto-run (OpenAction/AA), Launch, embedded files, "
                      "forms, obfuscated keywords")
    if text_status in ("ok", "partial"):
        checks.append(f"Text extraction ({text_chars:,} characters via {text_source}): phishing wording, "
                      "payment/credential requests, impersonation")
    checks.append(f"Links and URL wording ({len(urls)} found, never visited): look-alike domains, "
                  "shortened/IP/odd-TLD links")
    if PYMUPDF_AVAILABLE and mu["parse_ok"]:
        checks.append("Link text compared with link destination; form fields; annotations; hidden text")

    status = "complete"
    if issues:
        status = "failed" if blocked else "partial"

    js_count = counts["JS"] + counts["JavaScript"]
    return {
        "engine": engine,
        "file_size": file_size,
        "pages": pages,
        "encrypted": encrypted,
        "password_required": mu["password_required"],
        "metadata": mu["metadata"],
        "urls": urls,
        "embedded_files": embedded_count,
        "embedded_names": embedded_names,
        "dangerous_embedded_names": dangerous,
        "javascript": js_count > 0,
        "javascript_count": js_count,
        "open_action": counts["OpenAction"],
        "additional_actions": counts["AA"],
        "launch_action": counts["Launch"],
        "submit_form": counts["SubmitForm"],
        "import_data": counts["ImportData"],
        "rich_media": counts["RichMedia"],
        "xfa": counts["XFA"],
        "goto_remote": counts["GoToR"] + counts["GoToE"],
        "jbig2": counts["JBIG2Decode"],
        "obfuscated_names": obfuscated,
        "header_offset": header_offset,
        "parse_ok": not blocked or encryption_blocks,
        "object_counts": {n: counts[n] for n in SUSPICIOUS_OBJECT_NAMES if counts[n] > 0},
        "notes": notes,
        "text": text,
        "text_chars": text_chars,
        "text_source": text_source,
        "text_extraction": text_status,
        "image_only": image_only,
        "link_mismatches": list(mu["link_mismatches"]),
        "form_fields": form_fields,
        "annotation_types": dict(annot_types),
        "hidden_chars": mu["hidden_chars"],
        "repaired": mu["repaired"],
        "acroform": counts["AcroForm"],
        "file_attachment": counts["FileAttachment"],
        "encryption_blocks_analysis": encryption_blocks,
        "analysis_issues": issues,
        "analysis_status": status,
        "checks_performed": checks,
    }


_ML_MODEL = load_pdf_model()      # None unless pdf_model.pkl exists


def analyze_pdf(path):
    """
    Statically analyze the PDF at `path` and classify it.

    The caller (the Flask route) is responsible for deleting `path`
    afterwards in a `finally:` block.

    An unexpected failure while inspecting the file is reported as an
    INCONCLUSIVE / analysis-failed result - it is never turned into SAFE.
    """
    try:
        findings = extract_pdf_findings(path)
    except Exception as exc:
        print("PDF ANALYSIS ERROR:", repr(exc))        # details stay on the server
        size = os.path.getsize(path) if os.path.exists(path) else 0
        findings = _failed_findings(size, "The analysis failed unexpectedly while inspecting this PDF.")

    verdict = classify_pdf(findings, model=_ML_MODEL)

    suspicious_objects = [f"/{name} (found {count}x)" for name, count in findings["object_counts"].items()]
    suspicious_objects += [f"/{name} written with hex escapes (obfuscated)" for name in findings["obfuscated_names"]]

    return {
        "classification": verdict["classification"],
        "risk_score": verdict["risk_score"],
        "evidence_score": verdict["evidence_score"],
        "pages": findings["pages"],
        "file_size": findings["file_size"],
        "encrypted": findings["encrypted"],
        "metadata": findings["metadata"],
        "urls": findings["urls"],
        "embedded_files": findings["embedded_files"],
        "embedded_names": findings["embedded_names"],
        "javascript": findings["javascript"],
        "suspicious_objects": suspicious_objects,
        "reasons": verdict["reasons"],
        "notes": findings["notes"] + verdict["notes"],
        "detection_method": verdict["detection_method"],
        "analysis_engine": findings["engine"],
        "ml_used": verdict["ml_used"],
        "analysis_status": findings["analysis_status"],
        "text_chars": findings["text_chars"],
        "text_extraction": findings["text_extraction"],
        "form_fields": len(findings["form_fields"]),
        "checks_performed": findings["checks_performed"],
    }
