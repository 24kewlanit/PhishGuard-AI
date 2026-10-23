/*
 * test_email_pdf_frontend_harness.js
 * Runs the REAL static/email_pdf.js inside Node with a tiny DOM stub and a mocked
 * fetch(), feeding it API responses captured from the Flask backend, and prints
 * what the page would show. Used by test_email_pdf_accuracy.py (no browser needed).
 *   node test_email_pdf_frontend_harness.js scenarios.json
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

class El {
    constructor(tag, id) {
        this.tag = tag; this.id = id || ""; this.children = []; this._text = "";
        this.className = ""; this.style = {}; this.value = ""; this.files = []; this.disabled = false;
        this.listeners = {};
        const self = this;
        this.classList = {
            add(c) { if (!self.className.split(/\s+/).includes(c)) self.className = (self.className + " " + c).trim(); },
            remove(c) { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(" "); },
            contains(c) { return self.className.split(/\s+/).includes(c); },
        };
    }
    set textContent(v) { this._text = String(v); this.children = []; }
    get textContent() { return [this._text, ...this.children.map((c) => c.textContent)].filter(Boolean).join(" "); }
    appendChild(c) { this.children.push(c); return c; }
    replaceChildren() { this.children = []; this._text = ""; }
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
    find(cls) {
        for (const c of this.children) {
            if (c.className.split(/\s+/).includes(cls)) return c;
            const d = c.find(cls); if (d) return d;
        }
        return null;
    }
}

async function runScenario(src, sc) {
    const ids = ["epEmailFile", "epEmailText", "epPdfFile", "epAnalyze", "epClear", "epError", "epLoading", "epResults"];
    const els = {}; ids.forEach((i) => (els[i] = new El("x", i)));
    els.epResults.className = "ep-results hidden";
    els.epEmailText.value = "placeholder so client-side validation passes";
    const docListeners = {};
    const document = {
        getElementById: (id) => els[id] || null,
        createElement: (t) => new El(t),
        addEventListener: (t, fn) => { docListeners[t] = fn; },
    };
    const fetchMock = async () => ({
        status: sc.status,
        json: async () => { if (sc.body === null) throw new Error("not json"); return sc.body; },
    });
    class FormData { append() {} }
    const ctx = vm.createContext({ document, fetch: fetchMock, FormData, console, Math, String, Array, Object });
    vm.runInContext(src, ctx);
    docListeners.DOMContentLoaded();
    await els.epAnalyze.listeners.click[0]();

    const results = els.epResults;
    const visible = !results.classList.contains("hidden") && results.children.length > 0;
    const out = { name: sc.name, resultsVisible: visible, error: els.epError.textContent, overall: "", overallClass: "",
                  overallScore: "", emailText: "", pdfText: "", pdfBadge: "", pdfCardClass: "", emailBadge: "" };
    if (visible) {
        const box = results.children[0];
        out.overallClass = box.className;
        out.overall = (box.find("ep-overall-class") || { textContent: "" }).textContent;
        out.overallScore = (box.find("ep-score") || { textContent: "" }).textContent;
        const cards = results.children[1];
        for (const card of cards.children) {
            const title = (card.find("ep-card-title") || { textContent: "" }).textContent;
            const badge = (card.find("ep-badge") || { textContent: "" }).textContent;
            if (title === "EMAIL ANALYSIS") { out.emailText = card.textContent; out.emailBadge = badge; }
            if (title === "PDF ANALYSIS") { out.pdfText = card.textContent; out.pdfBadge = badge; out.pdfCardClass = card.className; }
        }
    }
    return out;
}

(async () => {
    const src = fs.readFileSync(path.join(__dirname, "static", "email_pdf.js"), "utf8");
    const scenarios = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const rows = [];
    for (const sc of scenarios) rows.push(await runScenario(src, sc));
    process.stdout.write(JSON.stringify(rows));
})().catch((e) => { console.error(e.stack || e); process.exit(1); });
