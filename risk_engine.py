import re
from urllib.parse import urlparse

# These are only well-known examples. They NEVER override strong phishing
# evidence or high ML phishing probability.
TRUSTED_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "github.com",
    "linkedin.com", "facebook.com", "instagram.com", "youtube.com",
    "wikipedia.org", "openai.com", "adobe.com", "dropbox.com", "netflix.com",
    "spotify.com", "zoom.us", "reddit.com", "paypal.com",
    "sbi.co.in", "rbi.org.in", "icicibank.com", "hdfcbank.com",
    "axisbank.com", "bankofbaroda.in", "npci.org.in", "uidai.gov.in",
}


BRAND_DOMAINS = {
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "paypal": "paypal.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "youtube": "youtube.com",
    "github": "github.com",
    "netflix": "netflix.com",
    "sbi": "sbi.co.in",
    "icici": "icicibank.com",
    "hdfc": "hdfcbank.com",
    "axis": "axisbank.com",
    "rbi": "rbi.org.in",
    "npci": "npci.org.in",
    "uidai": "uidai.gov.in",
}


SUSPICIOUS_HOST_WORDS = {
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "unlock", "suspend",
    "banking", "wallet", "payment", "authenticate", "auth", "password",
}


def registered_domain(hostname):
    host = (hostname or "").lower().strip(".")
    if not host:
        return ""

    # Common multi-label public suffixes. This is intentionally dependency-free.
    multi_suffixes = {
        "co.uk", "org.uk", "ac.uk", "gov.uk",
        "com.au", "net.au", "org.au",
        "co.in", "firm.in", "net.in", "org.in", "gen.in",
        "co.jp", "ne.jp", "or.jp",
        "co.nz", "com.br", "com.cn", "com.sg",
    }

    parts = host.split(".")
    if len(parts) < 2:
        return host

    suffix = ".".join(parts[-2:])
    if suffix in multi_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def hostname_from_url(url):
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except Exception:
        return ""


def is_trusted_domain(url):
    return registered_domain(hostname_from_url(url)) in TRUSTED_DOMAINS


def _brand_mismatch(hostname):
    """Return brands impersonated by the hostname, if any.

    Example:
      secure-google-login.example.com -> google
      google.example.com -> google
    but:
      google.com -> no mismatch
    """
    host = hostname_from_url(hostname)
    if not host:
        return []

    reg = registered_domain(host)
    tokens = set(re.findall(r"[a-z0-9]+", host.replace("-", ".")))

    found = []
    for brand, official in BRAND_DOMAINS.items():
        official_reg = registered_domain(official)
        if brand in tokens and reg != official_reg:
            found.append(brand)
    return found


def calculate_risk_score(phishing_probability, website=None, features=None, url=""):
    """Hybrid phishing decision layer.

    The Random Forest is the main ML signal. Browser behaviour adds evidence
    such as cross-domain credential forms, redirects, external iframes and
    suspicious navigation. Exact trusted domains reduce weak false positives,
    but can NEVER erase strong phishing evidence.
    """
    phishing_probability = max(0.0, min(float(phishing_probability), 100.0))
    website = website or {}
    features = features

    def f(name, default=0.0):
        try:
            return float(features[name].iloc[0])
        except Exception:
            return default

    hostname = hostname_from_url(url)
    reg_domain = registered_domain(hostname)
    trusted = reg_domain in TRUSTED_DOMAINS

    # Start from ML probability, but keep a meaningful floor so a few
    # browser signals can move borderline cases.
    score = 0.88 * phishing_probability
    reasons = []
    strong_signals = 0

    def add(points, reason, strong=False):
        nonlocal score, strong_signals
        score += points
        reasons.append(reason)
        if strong:
            strong_signals += 1

    # ---------------- URL evidence ----------------
    if f("IsDomainIP") == 1:
        add(22, "The URL uses an IP address instead of a normal domain.", True)

    if "xn--" in hostname:
        add(18, "The domain uses punycode, which can be used for look-alike domains.", True)

    if f("NoOfObfuscatedChar") >= 2:
        add(12, "The URL contains encoded or obfuscated characters.", True)

    if f("NoOfSubDomain") >= 4:
        add(8, "The URL contains an unusually deep subdomain structure.", True)

    if f("URLLength") >= 150:
        add(9, "The URL is unusually long.")
    elif f("URLLength") >= 100:
        add(4, "The URL is relatively long.")

    if f("IsHTTPS") == 0:
        add(5, "The URL does not use HTTPS.")

    # Hyphen-heavy hostnames are not phishing by themselves, but are useful
    # when combined with brand impersonation or login/verification words.
    hyphens = hostname.count("-")
    if hyphens >= 3:
        add(4, "The hostname contains several hyphens.")

    # ---------------- Hostname impersonation ----------------
    impersonated = _brand_mismatch(hostname)
    if impersonated:
        names = ", ".join(sorted(impersonated))
        add(25, f"The hostname appears to imitate {names} but is not the official domain.", True)

    host_tokens = set(re.findall(r"[a-z0-9]+", hostname.replace("-", ".")))
    risky_host_words = sorted(host_tokens.intersection(SUSPICIOUS_HOST_WORDS))

    if risky_host_words and not trusted:
        # Generic words alone are not enough. They become useful when the ML
        # score is already non-trivial or the host looks like impersonation.
        if phishing_probability >= 45 or impersonated:
            add(min(10, 3 * len(risky_host_words)),
                "The hostname contains security-sensitive words such as "
                + ", ".join(risky_host_words) + ".")

    # ---------------- Browser behaviour ----------------
    external_forms = int(website.get("external_form_count", 0) or 0)
    otp = int(website.get("otp_count", 0) or 0)
    password = int(website.get("password_count", 0) or 0)
    redirects = int(website.get("redirect_count", 0) or 0)
    external_iframes = int(website.get("external_iframe_count", 0) or 0)
    external_requests = int(website.get("external_request_count", 0) or 0)
    download_count = int(website.get("download_count", 0) or 0)
    popup_count = int(website.get("popup_count", 0) or 0)
    final_url = website.get("final_url", url) or url

    if external_forms:
        add(min(external_forms * 20, 40),
            "A form submits information to a different domain.", True)

    if otp and external_forms:
        add(10, "An OTP field is combined with an external form destination.", True)

    if password and external_forms:
        add(8, "A password field submits toward an external domain.", True)

    if redirects >= 4:
        add(6, f"The page uses {redirects} redirects.")

    if external_iframes >= 4:
        add(5, f"The page contains {external_iframes} external iframes.")

    if external_requests >= 25:
        add(3, f"The page contacted many external domains ({external_requests} requests).")

    if download_count:
        add(4, "The page triggered a file download.")

    if popup_count >= 2:
        add(2, f"The page opened {popup_count} additional browser pages.")

    # Redirecting to a different registrable domain is more informative than
    # simply having a few normal redirects.
    final_reg = registered_domain(hostname_from_url(final_url))
    if final_reg and reg_domain and final_reg != reg_domain:
        add(15, "The website redirected to a different registered domain.", True)

    # High-risk credential terms are only meaningful with another strong signal.
    keywords = {str(k).lower() for k in (website.get("suspicious_keywords") or [])}
    high_risk_terms = keywords.intersection({"cvv", "one time password", "credential", "credentials"})
    if high_risk_terms and (external_forms or impersonated or f("IsDomainIP") == 1):
        add(min(10, 5 * len(high_risk_terms)),
            "Sensitive credential terms appear alongside stronger phishing indicators.", True)

    # ---------------- Trusted-domain handling ----------------
    # NEVER force a trusted domain to SAFE. Compromised pages can exist on
    # legitimate domains, and paths can still be malicious.
    if trusted and f("IsHTTPS") == 1 and phishing_probability < 25 and strong_signals == 0:
        score = min(score, 10)
        reasons.insert(0, "The registered domain matches a configured well-known domain and uses HTTPS; weak page terms are treated as contextual.")
    elif trusted and f("IsHTTPS") == 1 and phishing_probability < 45 and strong_signals <= 1:
        score = min(score, 30)
        reasons.insert(0, "The registered domain matches a configured well-known domain; only weak additional risk signals were found.")

    score = max(0, min(round(score), 100))

    if score >= 70:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    elif score >= 20:
        level = "LOW"
    else:
        level = "SAFE"

    if not reasons:
        reasons.append("No strong phishing indicators were detected.")

    return score, level, list(dict.fromkeys(reasons))
