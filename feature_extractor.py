import ipaddress
import re
from urllib.parse import urlparse
import pandas as pd

FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "IsHTTPS",
    "TLDLength",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "LetterRatioInURL",
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
]

SPECIAL_CHARS = set("!#$%&'()*+,;:<=>?@[\\]^_`{|}~")


def normalize_url(url):
    url = str(url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url
    return url


def extract_features(url):
    url = normalize_url(url)
    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    domain = hostname
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    parts = hostname.split(".") if hostname else []
    no_of_subdomain = max(len(parts) - 2, 0)
    tld_length = len(parts[-1]) if parts and not is_domain_ip else 0

    url_length = len(url)
    domain_length = len(domain)
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    equals = url.count("=")
    qmarks = url.count("?")
    ampersands = url.count("&")

    other_special = sum(
        c in SPECIAL_CHARS and c not in "=?" + "&"
        for c in url
    )

    obfuscated = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    special_total = sum(not c.isalnum() for c in url)

    return pd.DataFrame([{
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "NoOfSubDomain": no_of_subdomain,
        "NoOfLettersInURL": letters,
        "NoOfDegitsInURL": digits,
        "NoOfEqualsInURL": equals,
        "NoOfQMarkInURL": qmarks,
        "NoOfAmpersandInURL": ampersands,
        "NoOfOtherSpecialCharsInURL": other_special,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
        "TLDLength": tld_length,
        "NoOfObfuscatedChar": obfuscated,
        "ObfuscationRatio": obfuscated / max(url_length, 1),
        "LetterRatioInURL": letters / max(url_length, 1),
        "DegitRatioInURL": digits / max(url_length, 1),
        "SpacialCharRatioInURL": special_total / max(url_length, 1),
    }], columns=FEATURE_NAMES)
