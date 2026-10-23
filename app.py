import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from urllib.parse import urlparse

import joblib
from flask import Flask, jsonify, render_template, request, url_for

from feature_extractor import extract_features, normalize_url
from risk_engine import calculate_risk_score, is_trusted_domain
from web_analyzer import analyze_website

# --- NEW: Email + PDF scanner (separate module, does not touch the URL scanner) ---
from email_pdf_routes import email_pdf_bp

app = Flask(__name__)

# NEW: hard cap on any request body (largest allowed upload is the 10 MB PDF).
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.register_blueprint(email_pdf_bp)   # adds POST /email-pdf-scan

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "static", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

MAX_HISTORY_ITEMS = 10
WEBSITE_ANALYSIS_TIMEOUT_SECONDS = 25
scan_history = []
executor = ThreadPoolExecutor(max_workers=2)

try:
    model = joblib.load(MODEL_PATH)
    print("ML model loaded.")
except Exception as e:
    model = None
    print("Model load failed:", e)


def is_valid_url(url):
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.hostname)
    except Exception:
        return False


def empty_website_result(url, message):
    return {
        "final_url": url,
        "title": "",
        "load_error": message,
        "screenshot_saved": False,
        "suspicious_keywords": [],
        "form_count": 0,
        "input_count": 0,
        "username_count": 0,
        "password_count": 0,
        "otp_count": 0,
        "external_form_count": 0,
        "redirect_count": 0,
        "redirect_chain": [],
        "link_count": 0,
        "external_link_count": 0,
        "same_domain_link_count": 0,
        "iframe_count": 0,
        "external_iframe_count": 0,
        "script_count": 0,
        "external_script_count": 0,
        "inline_script_count": 0,
    }


def run_website_analysis(url, screenshot_path):
    future = executor.submit(analyze_website, url, screenshot_path)
    try:
        return future.result(timeout=WEBSITE_ANALYSIS_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        return empty_website_result(
            url,
            f"Browser analysis timed out after {WEBSITE_ANALYSIS_TIMEOUT_SECONDS} seconds."
        )
    except Exception as e:
        return empty_website_result(url, f"Browser analysis failed: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify(success=False, error="ML model is not available."), 500

        data = request.get_json(silent=True) or {}
        raw_url = str(data.get("url", "")).strip()

        if not raw_url:
            return jsonify(success=False, error="Please enter a URL.")

        url = normalize_url(raw_url)

        if not is_valid_url(url):
            return jsonify(success=False, error="Please enter a valid HTTP/HTTPS URL.")

        features = extract_features(url)

        expected = list(getattr(model, "feature_names_in_", features.columns))
        missing = [x for x in expected if x not in features.columns]
        if missing:
            return jsonify(
                success=False,
                error="Model/feature mismatch: " + ", ".join(missing)
            ), 500

        features = features[expected]

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        # PhiUSIIL dataset label meaning:
        # 0 = phishing
        # 1 = legitimate
        if 0 not in classes or 1 not in classes:
            return jsonify(success=False, error="Model must contain labels 0 and 1."), 500

        phishing_probability = float(probabilities[classes.index(0)]) * 100
        legitimate_probability = float(probabilities[classes.index(1)]) * 100

        ml_result = "PHISHING" if prediction == 0 else "LEGITIMATE"
        ml_confidence = (
            phishing_probability if prediction == 0
            else legitimate_probability
        )

        scan_id = uuid.uuid4().hex[:12]
        screenshot_filename = f"{scan_id}.png"
        screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_filename)

        website = run_website_analysis(url, screenshot_path)

        screenshot_url = None
        if website.get("screenshot_saved") and os.path.exists(screenshot_path):
            screenshot_url = url_for(
                "static",
                filename=f"screenshots/{screenshot_filename}"
            )

        risk_score, risk_level, reasons = calculate_risk_score(
            phishing_probability=phishing_probability,
            website=website,
            features=features,
            url=url
        )

        if risk_score >= 70:
            final_result = "PHISHING"
            safe = False
        elif risk_score >= 40:
            final_result = "SUSPICIOUS"
            safe = False
        else:
            final_result = "LEGITIMATE"
            safe = True

        clean_features = {}
        for key, value in features.iloc[0].to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            clean_features[key] = value

        history_item = {
            "scan_id": scan_id,
            "url": url,
            "result": final_result,
            "ml_result": ml_result,
            "ml_confidence": round(ml_confidence, 2),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        scan_history.insert(0, history_item)
        del scan_history[MAX_HISTORY_ITEMS:]

        return jsonify({
            "success": True,
            "url": url,
            "final_url": website.get("final_url", url),
            "title": website.get("title", ""),
            "screenshot_url": screenshot_url,
            "load_error": website.get("load_error"),
            "redirect_chain": website.get("redirect_chain", []),

            "ml_result": ml_result,
            "ml_confidence": round(ml_confidence, 2),
            "phishing_probability": round(phishing_probability, 2),
            "legitimate_probability": round(legitimate_probability, 2),

            "result": final_result,
            "safe": safe,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": reasons,
            "domain_trusted": is_trusted_domain(url),

            "suspicious_keywords": website.get("suspicious_keywords", []),
            "form_count": website.get("form_count", 0),
            "input_count": website.get("input_count", 0),
            "username_count": website.get("username_count", 0),
            "password_count": website.get("password_count", 0),
            "otp_count": website.get("otp_count", 0),
            "external_form_count": website.get("external_form_count", 0),
            "redirect_count": website.get("redirect_count", 0),
            "link_count": website.get("link_count", 0),
            "external_link_count": website.get("external_link_count", 0),
            "same_domain_link_count": website.get("same_domain_link_count", 0),
            "iframe_count": website.get("iframe_count", 0),
            "external_iframe_count": website.get("external_iframe_count", 0),
            "script_count": website.get("script_count", 0),
            "external_script_count": website.get("external_script_count", 0),
            "inline_script_count": website.get("inline_script_count", 0),
            "external_request_count": website.get("external_request_count", 0),
            "unique_request_domains": website.get("unique_request_domains", 0),
            "download_count": website.get("download_count", 0),
            "popup_count": website.get("popup_count", 0),

            "features": clean_features
        })

    except Exception as e:
        print("SCAN ERROR:", repr(e))
        return jsonify(
            success=False,
            error="Unable to analyze the URL: " + str(e)
        ), 500


@app.route("/history")
def history():
    return jsonify(success=True, history=scan_history)


@app.route("/health")
def health():
    return jsonify(
        status="online",
        model_loaded=model is not None,
        service="PhishGuard AI"
    )


if __name__ == "__main__":
    print("\nPHISHGUARD AI")
    print("ML Model:", "ONLINE" if model is not None else "FAILED")
    app.run(host="127.0.0.1", port=5000, debug=False)
