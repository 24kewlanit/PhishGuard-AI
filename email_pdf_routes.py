"""
email_pdf_routes.py
===================
STAGE 8 + 9 - FLASK API for the Email + PDF scanner.

This is a Flask "Blueprint": a self-contained group of routes that app.py
registers with two lines. It adds ONE endpoint and does not touch /predict,
/history or /health.

    POST /email-pdf-scan      (multipart/form-data)
        email_file : an .eml file             (optional)
        email_text : pasted email content     (optional; used if no email_file)
        pdf_file   : a .pdf attachment        (optional)
      At least one of email_file / email_text / pdf_file is required.

Safety rules implemented here
-----------------------------
  * File extensions are validated (.eml / .pdf).
  * Size limits: .eml 5 MB, .pdf 10 MB, pasted text 200,000 characters.
  * A PDF must start with the "%PDF-" magic bytes.
  * The uploaded PDF is saved ONLY under a random temp name (the user's
    filename is never used as a path -> no path traversal).
  * The temp PDF is deleted in a `finally:` block, even if analysis crashes.
    Only the server's temp COPY is deleted - never the user's own file.
  * Errors are returned as short JSON messages; stack traces and server
    paths are never sent to the browser.
  * Nothing is executed, opened in a browser, or sent to a third party.
    URLs found in the email/PDF are only listed, never visited.
"""

import os
import tempfile
import time

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from combined_analyzer import combine_results
from email_analyzer import parse_eml_bytes, parse_email_text
from email_model import load_email_model, predict_email
import pdf_analyzer as _pdf_analyzer
from pdf_analyzer import PYMUPDF_AVAILABLE, analyze_pdf

email_pdf_bp = Blueprint("email_pdf", __name__)

# ---- limits ----
MAX_EML_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_EMAIL_TEXT_CHARS = 200_000
MAX_URLS_RETURNED = 50

# Dedicated temp folder (easy to inspect / verify that it is empty after a scan)
PDF_TEMP_DIR = os.path.join(tempfile.gettempdir(), "phishguard_pdf_tmp")
os.makedirs(PDF_TEMP_DIR, exist_ok=True)

# Load the email model once at start-up (None => rule-based fallback is used)
EMAIL_MODEL = load_email_model()
print("Email detector:", "ML model loaded" if EMAIL_MODEL is not None
      else "no email_model.pkl found -> rule-based heuristics")
print("PDF analyzer  :", "PyMuPDF available" if PYMUPDF_AVAILABLE
      else "PyMuPDF NOT installed -> built-in scanner only (less complete)",
      "| PDF ML model:", "loaded" if _pdf_analyzer._ML_MODEL is not None else "none (rules only)")


class ScanInputError(Exception):
    """A problem with the user's input. The message is safe to show to the user."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _purge_stale_temp_files(max_age_seconds=3600):
    """Remove leftovers from a previous crash (older than 1 hour)."""
    now = time.time()
    try:
        for name in os.listdir(PDF_TEMP_DIR):
            path = os.path.join(PDF_TEMP_DIR, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
                os.remove(path)
    except OSError:
        pass


_purge_stale_temp_files()


def _read_upload(file_storage, max_bytes, label):
    """Read an uploaded file into memory, refusing anything bigger than max_bytes."""
    data = file_storage.stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ScanInputError(f"The {label} is too large (limit {max_bytes // (1024 * 1024)} MB).", 413)
    if not data:
        raise ScanInputError(f"The {label} is empty.")
    return data


def _extension(filename):
    return os.path.splitext(filename or "")[1].lower()


@email_pdf_bp.route("/email-pdf-scan", methods=["POST"])
def email_pdf_scan():
    temp_pdf_path = None          # path of the server-side temporary copy
    try:
        email_file = request.files.get("email_file")
        pdf_file = request.files.get("pdf_file")
        email_text = (request.form.get("email_text") or "").strip()

        has_email_file = bool(email_file and email_file.filename)
        has_pdf = bool(pdf_file and pdf_file.filename)

        if not (has_email_file or email_text or has_pdf):
            raise ScanInputError("Please upload an .eml file, paste the email text, or upload a PDF.")
        if len(email_text) > MAX_EMAIL_TEXT_CHARS:
            raise ScanInputError(f"Pasted email is too long (limit {MAX_EMAIL_TEXT_CHARS:,} characters).", 413)

        # -------------------- EMAIL --------------------
        email_out, email_verdict = None, None
        if has_email_file or email_text:
            if has_email_file:
                if _extension(email_file.filename) != ".eml":
                    raise ScanInputError("Email upload must be a .eml file.")
                parsed = parse_eml_bytes(_read_upload(email_file, MAX_EML_BYTES, "email file"))
            else:
                parsed = parse_email_text(email_text)

            # Truly empty input is an input error. Unreadable/garbled input is NOT: it is analysed
            # and reported INCONCLUSIVE, and an email that only has attachments is still analysed.
            if parsed.get("parse_ok", True) and not (
                    parsed["body"].strip() or parsed["subject"].strip() or parsed["urls"] or parsed["attachments"]):
                raise ScanInputError("The email appears to be empty - no subject, body, links or attachments were found.")

            email_verdict = predict_email(parsed, EMAIL_MODEL)
            email_out = {
                "sender": parsed["sender"],
                "recipient": parsed["recipient"],
                "subject": parsed["subject"],
                "urls": parsed["urls"][:MAX_URLS_RETURNED],
                "url_count": len(parsed["urls"]),
                "attachments": parsed["attachments"],
                "classification": email_verdict["classification"],
                "risk_score": email_verdict["risk_score"],          # None when INCONCLUSIVE
                "evidence_score": email_verdict["evidence_score"],
                "reasons": email_verdict["reasons"],
                "notes": email_verdict["notes"],
                "checks_performed": email_verdict["checks_performed"],
                "analysis_status": email_verdict["analysis_status"],
                "detection_method": email_verdict["detection_method"],
                "ml_used": email_verdict["ml_used"],
            }

        # -------------------- PDF --------------------
        pdf_out, pdf_verdict = None, None
        if has_pdf:
            if _extension(pdf_file.filename) != ".pdf":
                raise ScanInputError("PDF upload must be a .pdf file.")
            pdf_bytes = _read_upload(pdf_file, MAX_PDF_BYTES, "PDF file")
            if b"%PDF-" not in pdf_bytes[:1024]:
                raise ScanInputError("This file is not a valid PDF (missing PDF signature).")

            # Save under a RANDOM name in our temp folder (never the user's filename).
            fd, temp_pdf_path = tempfile.mkstemp(prefix="pg_", suffix=".pdf", dir=PDF_TEMP_DIR)
            with os.fdopen(fd, "wb") as handle:
                handle.write(pdf_bytes)
            del pdf_bytes

            result = analyze_pdf(temp_pdf_path)        # static analysis - never executed
            pdf_verdict = result
            pdf_out = {
                "filename": secure_filename(pdf_file.filename) or "attachment.pdf",
                "classification": result["classification"],
                "risk_score": result["risk_score"],
                "pages": result["pages"],
                "file_size": result["file_size"],
                "javascript": result["javascript"],
                "embedded_files": result["embedded_files"],
                "embedded_names": result["embedded_names"],
                "encrypted": result["encrypted"],
                "urls": result["urls"][:MAX_URLS_RETURNED],
                "url_count": len(result["urls"]),
                "metadata": result["metadata"],
                "suspicious_objects": result["suspicious_objects"],
                "reasons": result["reasons"],
                "notes": result["notes"],
                "detection_method": result["detection_method"],
                "analysis_engine": result["analysis_engine"],
                "ml_used": result["ml_used"],
                "evidence_score": result["evidence_score"],
                "analysis_status": result["analysis_status"],
                "text_chars": result["text_chars"],
                "text_extraction": result["text_extraction"],
                "form_fields": result["form_fields"],
                "checks_performed": result["checks_performed"],
            }

        # -------------------- COMBINE --------------------
        overall = combine_results(email_verdict, pdf_verdict)

        return jsonify({
            "success": True,
            "email": email_out,
            "pdf": pdf_out,
            "overall": overall,
        })

    except ScanInputError as err:
        return jsonify(success=False, error=err.message), err.status

    except HTTPException as exc:
        # e.g. 413 when the whole request exceeds Flask's MAX_CONTENT_LENGTH
        message = "The upload is too large." if exc.code == 413 else "The request could not be processed."
        return jsonify(success=False, error=message), exc.code or 400

    except Exception as exc:
        # Log details on the SERVER only; send a generic message to the browser.
        print("EMAIL/PDF SCAN ERROR:", repr(exc))
        return jsonify(success=False,
                       error="The scan could not be completed. The file may be damaged or unsupported."), 500

    finally:
        # ALWAYS delete the server-side temporary PDF (success, validation error or crash).
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError as exc:
                print("Could not delete temp file:", repr(exc))
