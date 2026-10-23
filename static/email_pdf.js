/* =========================================================
   EMAIL + PDF SECURITY SCANNER  (new - separate from script.js)

   SECURITY: all text coming from the server (sender, subject, URLs, file
   names, reasons ...) is inserted with textContent, never innerHTML, so a
   malicious email cannot inject HTML/JS into this page.
========================================================= */
"use strict";

(function () {

    const MAX_EML_BYTES = 5 * 1024 * 1024;
    const MAX_PDF_BYTES = 10 * 1024 * 1024;

    const $ = (id) => document.getElementById(id);

    /* ---------- tiny DOM helpers ---------- */
    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = text;
        return node;
    }

    function statusClass(classification) {
        const c = String(classification || "").toUpperCase();
        if (c === "SAFE") return "ep-safe";
        if (c === "SPAM") return "ep-spam";
        if (c === "SUSPICIOUS") return "ep-suspicious";
        if (c === "PHISHING") return "ep-phishing";
        if (c === "POTENTIALLY MALICIOUS") return "ep-malicious";
        if (c === "INCONCLUSIVE") return "ep-inconclusive";
        return "ep-unknown";
    }

    function statusIcon(classification) {
        const c = String(classification || "").toUpperCase();
        if (c === "SAFE") return "✓ ";
        if (c === "SPAM") return "✉ ";
        if (c === "SUSPICIOUS") return "⚠ ";
        if (c === "INCONCLUSIVE") return "? ";
        return "✖ ";
    }

    /* A risk score of null means "no reliable score" (INCONCLUSIVE) - never show it as 0. */
    function hasScore(v) {
        return typeof v === "number" && isFinite(v);
    }

    function scoreText(v) {
        return hasScore(v) ? String(v) + "/100" : "N/A";
    }

    /* The API response must have the expected shape. If not, NOTHING is shown as a result
       (a missing field must never be displayed as "Safe"). */
    const KNOWN_CLASSES = ["SAFE", "SPAM", "SUSPICIOUS", "PHISHING", "POTENTIALLY MALICIOUS", "INCONCLUSIVE"];

    function validPart(part) {
        return part && typeof part === "object" &&
            KNOWN_CLASSES.indexOf(String(part.classification).toUpperCase()) !== -1 &&
            Array.isArray(part.reasons) &&
            (part.risk_score === null || hasScore(part.risk_score));
    }

    function validResponse(data) {
        if (!data || data.success !== true) return false;
        if (!validPart(data.overall) || !data.overall.breakdown || !data.overall.breakdown.weights) return false;
        if (data.email && !validPart(data.email)) return false;
        if (data.pdf && !validPart(data.pdf)) return false;
        return !!(data.email || data.pdf);
    }

    function formatBytes(n) {
        if (n < 1024) return n + " B";
        if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
        return (n / (1024 * 1024)).toFixed(1) + " MB";
    }

    function row(label, value) {
        const r = el("div", "ep-row");
        r.appendChild(el("span", null, label));
        r.appendChild(el("span", null, value === "" || value === undefined ? "—" : String(value)));
        return r;
    }

    function listBlock(title, items, extraClass, emptyText) {
        const wrap = el("div");
        wrap.appendChild(el("div", "ep-sub", title));
        if (!items || !items.length) {
            wrap.appendChild(el("div", "ep-none", emptyText || "None"));
            return wrap;
        }
        const list = el("div", "ep-list");
        items.forEach((t) => list.appendChild(el("div", "ep-item " + (extraClass || ""), t)));
        wrap.appendChild(list);
        return wrap;
    }

    /* ---------- rendering ---------- */
    function renderOverall(overall) {
        const box = el("div", "ep-overall " + statusClass(overall.classification));

        const left = el("div");
        left.appendChild(el("div", "ep-overall-label", "OVERALL RESULT"));
        left.appendChild(el("div", "ep-overall-class", statusIcon(overall.classification) + overall.classification));

        const right = el("div", "ep-score");
        const big = el("div");
        if (hasScore(overall.risk_score)) {
            big.appendChild(el("strong", null, String(overall.risk_score)));
            big.appendChild(el("span", null, " / 100 risk score"));
        } else {
            big.appendChild(el("strong", null, "N/A"));
            big.appendChild(el("span", null, " no reliable risk score (analysis incomplete)"));
        }
        right.appendChild(big);
        const meter = el("div", "ep-meter");
        const fill = el("div");
        fill.style.width = (hasScore(overall.risk_score) ? Math.max(0, Math.min(100, overall.risk_score)) : 0) + "%";
        meter.appendChild(fill);
        right.appendChild(meter);

        box.appendChild(left);
        box.appendChild(right);
        return box;
    }

    function renderEmail(email) {
        const card = el("div", "ep-card " + statusClass(email.classification));
        const head = el("div", "ep-card-head");
        head.appendChild(el("div", "ep-card-title", "EMAIL ANALYSIS"));
        head.appendChild(el("span", "ep-badge", email.classification + " · " + scoreText(email.risk_score)));
        card.appendChild(head);

        const rows = el("div", "ep-rows");
        rows.appendChild(row("Sender", email.sender));
        rows.appendChild(row("Subject", email.subject));
        rows.appendChild(row("URLs found", email.url_count));
        if (email.attachments && email.attachments.length) {
            rows.appendChild(row("Attachments in email",
                email.attachments.map((a) => a.filename).join(", ")));
        }
        card.appendChild(rows);

        card.appendChild(listBlock("URLS (NOT VISITED)", email.urls, "url", "No URLs found"));
        rows.appendChild(row("Analysis", email.analysis_status || "unknown"));
        card.appendChild(listBlock("REASONS", email.reasons, "", "No suspicious indicators detected"));
        card.appendChild(listBlock("CHECKS PERFORMED", email.checks_performed, "", "None"));
        (email.notes || []).forEach((n) => card.appendChild(el("div", "ep-note", "ℹ " + n)));
        card.appendChild(el("div", "ep-method", "Method: " + email.detection_method +
            ". The score is a rule-based evidence score, not a probability. SAFE only means no indicator " +
            "was found by the checks that could be run."));
        return card;
    }

    function renderPdf(pdf) {
        const card = el("div", "ep-card " + statusClass(pdf.classification));
        const head = el("div", "ep-card-head");
        head.appendChild(el("div", "ep-card-title", "PDF ANALYSIS"));
        head.appendChild(el("span", "ep-badge", pdf.classification + " · " + scoreText(pdf.risk_score)));
        card.appendChild(head);

        const rows = el("div", "ep-rows");
        rows.appendChild(row("Filename", pdf.filename));
        rows.appendChild(row("Pages", pdf.pages || "unknown"));
        rows.appendChild(row("File size", formatBytes(pdf.file_size)));
        rows.appendChild(row("JavaScript", pdf.javascript ? "Detected" : "Not detected"));
        rows.appendChild(row("Embedded files", pdf.embedded_files));
        rows.appendChild(row("Encrypted", pdf.encrypted ? "Yes" : "No"));
        rows.appendChild(row("URLs found", pdf.url_count));
        rows.appendChild(row("Text extracted", (pdf.text_chars || 0) + " characters (" + (pdf.text_extraction || "unknown") + ")"));
        rows.appendChild(row("Form fields", pdf.form_fields || 0));
        rows.appendChild(row("Analysis", pdf.analysis_status || "unknown"));
        card.appendChild(rows);

        card.appendChild(listBlock("SUSPICIOUS OBJECTS", pdf.suspicious_objects, "", "None found"));
        card.appendChild(listBlock("URLS (NOT VISITED)", pdf.urls, "url", "No URLs found"));
        card.appendChild(listBlock("REASONS", pdf.reasons, "", "No suspicious indicators detected"));
        card.appendChild(listBlock("CHECKS PERFORMED", pdf.checks_performed, "", "None"));

        (pdf.notes || []).forEach((n) => card.appendChild(el("div", "ep-note", "ℹ " + n)));
        card.appendChild(el("div", "ep-method",
            "Method: " + pdf.detection_method + ". The PDF was analysed statically (parsed, never executed) " +
            "and the temporary server copy was deleted. The score is a rule-based evidence score, not a " +
            "probability. SAFE only means no indicator was found by the checks that could be run."));
        return card;
    }

    function renderResults(data) {
        const out = $("epResults");
        out.replaceChildren();

        out.appendChild(renderOverall(data.overall));

        const cards = el("div", "ep-cards");
        if (data.email) cards.appendChild(renderEmail(data.email));
        if (data.pdf) cards.appendChild(renderPdf(data.pdf));
        if (!data.email || !data.pdf) cards.style.gridTemplateColumns = "1fr";
        out.appendChild(cards);

        const why = el("div", "ep-card ep-overall-reasons " + statusClass(data.overall.classification));
        why.appendChild(el("div", "ep-card-title", "WHY THIS OVERALL RESULT"));
        const list = el("div", "ep-list");
        list.style.marginTop = "12px";
        // The detailed indicators are already listed in the Email / PDF cards above,
        // so this card only shows the override explanations (if any) and the formula.
        const overrides = data.overall.reasons.filter(
            (r) => !r.startsWith("Email: ") && !r.startsWith("PDF: ")
        );
        if (overrides.length) {
            overrides.forEach((r) => list.appendChild(el("div", "ep-item", r)));
        } else {
            list.appendChild(el("div", "ep-item",
                "No override rule was needed - the weighted score decided the result. " +
                "See the Email and PDF reasons above for the detected indicators."));
        }
        why.appendChild(list);
        const b = data.overall.breakdown;
        const w = b.weights;
        let formula = "Score = ";
        if (hasScore(b.email_risk) && hasScore(b.pdf_risk)) {
            formula += (w.email * 100) + "% × email (" + b.email_risk + ") + " +
                       (w.pdf * 100) + "% × PDF (" + b.pdf_risk + ") = " + b.weighted_score +
                       ", then raised by any override rule above.";
        } else if (hasScore(b.email_risk)) {
            formula += "100% × email (" + b.email_risk + ")" +
                       (data.pdf ? " (the PDF result was inconclusive and has no score)." : ".");
        } else if (hasScore(b.pdf_risk)) {
            formula += "100% × PDF (" + b.pdf_risk + ")" +
                       (data.email ? " (the email result was inconclusive and has no score)." : ".");
        } else {
            formula = "No reliable risk score could be calculated because the analysis was inconclusive.";
        }
        why.appendChild(el("div", "ep-method", formula));
        out.appendChild(why);

        out.classList.remove("hidden");
    }

    /* ---------- validation + submit ---------- */
    function validate(emailFile, emailText, pdfFile) {
        if (!emailFile && !emailText && !pdfFile) {
            return "Please upload an .eml file, paste the email text, or upload a PDF.";
        }
        if (emailFile) {
            if (!emailFile.name.toLowerCase().endsWith(".eml")) return "The email file must be a .eml file.";
            if (emailFile.size > MAX_EML_BYTES) return "The .eml file is too large (limit 5 MB).";
        }
        if (pdfFile) {
            if (!pdfFile.name.toLowerCase().endsWith(".pdf")) return "The attachment must be a .pdf file.";
            if (pdfFile.size > MAX_PDF_BYTES) return "The PDF is too large (limit 10 MB).";
        }
        return null;
    }

    async function analyze() {
        const emailFile = $("epEmailFile").files[0] || null;
        const emailText = $("epEmailText").value.trim();
        const pdfFile = $("epPdfFile").files[0] || null;

        const error = $("epError");
        const button = $("epAnalyze");
        const loading = $("epLoading");
        const results = $("epResults");

        error.textContent = "";
        results.classList.add("hidden");

        const problem = validate(emailFile, emailText, pdfFile);
        if (problem) { error.textContent = problem; return; }

        const form = new FormData();
        if (emailFile) form.append("email_file", emailFile);
        else if (emailText) form.append("email_text", emailText);
        if (pdfFile) form.append("pdf_file", pdfFile);

        button.disabled = true;
        loading.classList.remove("hidden");

        try {
            const response = await fetch("/email-pdf-scan", { method: "POST", body: form });
            let data = null;
            try { data = await response.json(); } catch (e) { /* non-JSON error page */ }

            if (!data || !data.success) {
                error.textContent = (data && data.error) || "The scan failed (HTTP " + response.status + ").";
                return;
            }
            if (!validResponse(data)) {
                // Never fall back to "Safe" when the response is incomplete or unexpected.
                error.textContent = "The server returned an incomplete or unexpected result, so no verdict " +
                                    "is shown. Please try again.";
                return;
            }
            renderResults(data);
        } catch (e) {
            error.textContent = "Unable to reach the server. Make sure Flask is running.";
        } finally {
            button.disabled = false;
            loading.classList.add("hidden");
        }
    }

    function clearAll() {
        $("epEmailFile").value = "";
        $("epEmailText").value = "";
        $("epPdfFile").value = "";
        $("epError").textContent = "";
        $("epResults").classList.add("hidden");
        $("epResults").replaceChildren();
    }

    document.addEventListener("DOMContentLoaded", function () {
        if (!$("epAnalyze")) return;
        $("epAnalyze").addEventListener("click", analyze);
        $("epClear").addEventListener("click", clearAll);
    });

})();
