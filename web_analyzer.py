from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright


def _same_registered_domain(a, b):
    def reg(url):
        host = (urlparse(url).hostname or "").lower().strip(".")
        parts = host.split(".")
        if len(parts) < 2:
            return host
        suffix = ".".join(parts[-2:])
        multi = {
            "co.uk", "org.uk", "ac.uk", "gov.uk",
            "com.au", "net.au", "org.au",
            "co.in", "net.in", "org.in",
            "co.jp", "co.nz", "com.br", "com.cn", "com.sg"
        }
        if suffix in multi and len(parts) >= 3:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])
    return bool(reg(a) and reg(a) == reg(b))


def analyze_website(url, screenshot_path="website_screenshot.png"):
    """Inspect a live website in an isolated, headless Chromium session.

    This is a lightweight sandbox-style layer: redirects, forms, credential
    fields, external destinations, requests, iframes, scripts, downloads and
    popups are observed without supplying credentials or interacting with
    forms.
    """
    results = {
        "final_url": url,
        "title": "",
        "load_error": None,
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
        "external_request_count": 0,
        "unique_request_domains": 0,
        "download_count": 0,
        "popup_count": 0,
        "is_domain_ip": 0,
    }

    redirect_chain = []
    request_domains = set()
    download_count = 0
    popup_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

        context = browser.new_context(
            ignore_https_errors=True,
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        def on_response(response):
            try:
                req_url = response.url
                host = urlparse(req_url).hostname
                if host:
                    request_domains.add(host.lower())
                if response.request.is_navigation_request() and response.status in (301, 302, 303, 307, 308):
                    redirect_chain.append(f"{response.status} -> {response.url}")
            except Exception:
                pass

        def on_download(download):
            nonlocal download_count
            download_count += 1

        def on_popup(popup):
            nonlocal popup_count
            popup_count += 1

        page.on("response", on_response)
        page.on("download", on_download)
        page.on("popup", on_popup)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # Give client-side applications a short window to render.
            try:
                page.wait_for_timeout(1200)
            except Exception:
                pass
        except Exception:
            results["load_error"] = (
                "The website could not be fully loaded. It may be offline, "
                "blocking automated browsers, or taking too long to respond."
            )

        results["final_url"] = page.url

        try:
            results["title"] = page.title()
        except Exception:
            results["title"] = ""

        try:
            page.screenshot(path=screenshot_path, full_page=True, timeout=10000)
            results["screenshot_saved"] = True
        except Exception:
            pass

        try:
            page_text = page.locator("body").inner_text(timeout=5000)
        except Exception:
            page_text = ""

        suspicious_keywords = [
            "login", "signin", "sign in", "password", "verify",
            "verification", "account", "otp", "one time password",
            "kyc", "pin", "cvv", "credit card", "debit card",
            "bank", "banking", "payment", "wallet", "recover",
            "unlock", "suspend", "confirm", "update", "credential"
        ]
        text_lower = page_text.lower()
        results["suspicious_keywords"] = [
            k for k in suspicious_keywords if k in text_lower
        ]

        # Forms and credential fields
        forms = page.locator("form")
        inputs = page.locator("input")
        try:
            form_count = forms.count()
        except Exception:
            form_count = 0
        try:
            input_count = inputs.count()
        except Exception:
            input_count = 0

        password_count = username_count = otp_count = 0

        for i in range(input_count):
            try:
                element = inputs.nth(i)
                input_type = (element.get_attribute("type") or "").lower()
                name = (element.get_attribute("name") or "").lower()
                placeholder = (element.get_attribute("placeholder") or "").lower()
                autocomplete = (element.get_attribute("autocomplete") or "").lower()

                if input_type == "password" or autocomplete in {"current-password", "new-password"}:
                    password_count += 1

                if (
                    input_type == "email"
                    or any(x in name for x in ("username", "user", "email"))
                    or any(x in placeholder for x in ("username", "user", "email"))
                    or autocomplete in {"username", "email"}
                ):
                    username_count += 1

                if any(x in name for x in ("otp", "one-time", "one_time")) or "otp" in placeholder:
                    otp_count += 1
            except Exception:
                continue

        results["form_count"] = form_count
        results["input_count"] = input_count
        results["password_count"] = password_count
        results["username_count"] = username_count
        results["otp_count"] = otp_count

        page_domain = urlparse(page.url).netloc
        external_form_count = 0

        for i in range(form_count):
            try:
                action = forms.nth(i).get_attribute("action")
                if not action:
                    continue
                action_url = urljoin(page.url, action)
                if (
                    urlparse(action_url).hostname
                    and not _same_registered_domain(page.url, action_url)
                ):
                    external_form_count += 1
            except Exception:
                continue

        results["external_form_count"] = external_form_count

        # Links
        links = page.locator("a")
        try:
            link_count = links.count()
        except Exception:
            link_count = 0

        external_link_count = same_domain_link_count = 0
        for i in range(link_count):
            try:
                href = links.nth(i).get_attribute("href")
                if not href:
                    continue
                full_url = urljoin(page.url, href)
                host = urlparse(full_url).hostname
                if host and not _same_registered_domain(page.url, full_url):
                    external_link_count += 1
                elif host:
                    same_domain_link_count += 1
            except Exception:
                continue

        results["link_count"] = link_count
        results["external_link_count"] = external_link_count
        results["same_domain_link_count"] = same_domain_link_count

        # Iframes
        iframes = page.locator("iframe")
        try:
            iframe_count = iframes.count()
        except Exception:
            iframe_count = 0

        external_iframe_count = 0
        for i in range(iframe_count):
            try:
                src = iframes.nth(i).get_attribute("src")
                if not src:
                    continue
                iframe_url = urljoin(page.url, src)
                host = urlparse(iframe_url).hostname
                if host and not _same_registered_domain(page.url, iframe_url):
                    external_iframe_count += 1
            except Exception:
                continue

        results["iframe_count"] = iframe_count
        results["external_iframe_count"] = external_iframe_count

        # Scripts
        scripts = page.locator("script")
        try:
            script_count = scripts.count()
        except Exception:
            script_count = 0

        external_script_count = inline_script_count = 0
        for i in range(script_count):
            try:
                src = scripts.nth(i).get_attribute("src")
                if src:
                    external_script_count += 1
                else:
                    inline_script_count += 1
            except Exception:
                continue

        results["script_count"] = script_count
        results["external_script_count"] = external_script_count
        results["inline_script_count"] = inline_script_count

        # Network activity observed while the page loaded.
        page_host = (urlparse(page.url).hostname or "").lower()
        external_domains = {
            h for h in request_domains if h and h != page_host
        }
        results["external_request_count"] = len(external_domains)
        results["unique_request_domains"] = len(request_domains)
        results["download_count"] = download_count
        results["popup_count"] = popup_count

        # IP-host signal
        try:
            import ipaddress
            ipaddress.ip_address(page_host)
            results["is_domain_ip"] = 1
        except Exception:
            results["is_domain_ip"] = 0

        context.close()
        browser.close()

    return results


if __name__ == "__main__":
    url = input("Enter website URL: ").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    result = analyze_website(url)
    print("\nFinal URL:", result["final_url"])
    print("Title:", result["title"])
    print("Suspicious keywords:", result["suspicious_keywords"])
    print("Forms:", result["form_count"])
    print("Password fields:", result["password_count"])
    print("OTP fields:", result["otp_count"])
    print("External forms:", result["external_form_count"])
    print("Redirects:", result["redirect_count"])
    print("External links:", result["external_link_count"])
    print("External iframes:", result["external_iframe_count"])
    print("External requests:", result["external_request_count"])
    print("Downloads:", result["download_count"])
    print("Popups:", result["popup_count"])
