import joblib
from urllib.parse import urlparse, urljoin

from playwright.sync_api import sync_playwright

from feature_extractor import extract_features
from risk_engine import calculate_risk_score


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("phishing_model.pkl")


# ============================================================
# MACHINE LEARNING ANALYSIS
# ============================================================

def analyze_with_ml(url):

    X = extract_features(url)

    # Make feature order exactly match model
    if hasattr(model, "feature_names_in_"):

        X = X[
            list(model.feature_names_in_)
        ]

    prediction = int(
        model.predict(X)[0]
    )

    probability = model.predict_proba(X)[0]


    # ========================================================
    # PhiUSIIL label mapping
    #
    # 0 = PHISHING
    # 1 = LEGITIMATE
    # ========================================================

    legitimate_index = list(
        model.classes_
    ).index(0)

    phishing_index = list(
        model.classes_
    ).index(1)


    legitimate_probability = (
        probability[
            legitimate_index
        ] * 100
    )

    phishing_probability = (
        probability[
            phishing_index
        ] * 100
    )


    if prediction == 0:

        result = "PHISHING"

        confidence = phishing_probability

    else:

        result = "LEGITIMATE"

        confidence = legitimate_probability


    return (
        result,
        confidence,
        X
    )


# ============================================================
# WEBSITE ANALYSIS
# ============================================================

def analyze_website(url):

    results = {}

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page()

        redirect_chain = []


        # ----------------------------------------------------
        # REDIRECT DETECTION
        # ----------------------------------------------------

        def handle_response(response):

            if response.request.is_navigation_request():

                if response.status in [
                    301,
                    302,
                    303,
                    307,
                    308
                ]:

                    redirect_chain.append(
                        f"{response.status} -> {response.url}"
                    )


        page.on(
            "response",
            handle_response
        )


        print("\nOpening website...")


        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

        except Exception as e:

            print(
                "\nWebsite could not be fully loaded."
            )

            print(
                "Reason:",
                e
            )


        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        results["final_url"] = page.url

        try:

            results["title"] = page.title()

        except:

            results["title"] = ""


        # ----------------------------------------------------
        # SCREENSHOT
        # ----------------------------------------------------

        try:

            page.screenshot(
                path="website_screenshot.png",
                full_page=True
            )

        except Exception as e:

            print(
                "Screenshot could not be captured:",
                e
            )


        # ----------------------------------------------------
        # PAGE TEXT
        # ----------------------------------------------------

        try:

            page_text = page.locator(
                "body"
            ).inner_text()

        except:

            page_text = ""


        results["page_text"] = page_text


        # ----------------------------------------------------
        # SUSPICIOUS KEYWORDS
        # ----------------------------------------------------

        suspicious_keywords = [

            "login",
            "signin",
            "sign in",
            "password",
            "verify",
            "verification",
            "account",
            "otp",
            "one time password",
            "kyc",
            "pin",
            "cvv",
            "credit card",
            "debit card",
            "bank",
            "banking",
            "payment",
            "wallet",
            "recover",
            "unlock"

        ]


        text_lower = page_text.lower()

        detected_keywords = []


        for keyword in suspicious_keywords:

            if keyword in text_lower:

                detected_keywords.append(
                    keyword
                )


        results["suspicious_keywords"] = (
            detected_keywords
        )


        # ----------------------------------------------------
        # FORM ANALYSIS
        # ----------------------------------------------------

        forms = page.locator("form")

        form_count = forms.count()


        inputs = page.locator("input")

        input_count = inputs.count()


        password_count = 0
        username_count = 0
        otp_count = 0


        for i in range(input_count):

            input_element = inputs.nth(i)


            input_type = (
                input_element.get_attribute(
                    "type"
                )
                or ""
            ).lower()


            input_name = (
                input_element.get_attribute(
                    "name"
                )
                or ""
            ).lower()


            input_placeholder = (
                input_element.get_attribute(
                    "placeholder"
                )
                or ""
            ).lower()


            if input_type == "password":

                password_count += 1


            if (

                input_type == "email"

                or "username" in input_name

                or "user" in input_name

                or "email" in input_name

                or "username" in input_placeholder

                or "email" in input_placeholder

            ):

                username_count += 1


            if (

                "otp" in input_name

                or "otp" in input_placeholder

                or "one time" in input_name

            ):

                otp_count += 1


        results["form_count"] = form_count

        results["input_count"] = input_count

        results["password_count"] = password_count

        results["username_count"] = username_count

        results["otp_count"] = otp_count


        # ----------------------------------------------------
        # FORM DESTINATION
        # ----------------------------------------------------

        page_domain = urlparse(
            page.url
        ).netloc


        external_form_count = 0


        for i in range(form_count):

            form = forms.nth(i)

            action = form.get_attribute(
                "action"
            )


            if action:

                action_url = urljoin(
                    page.url,
                    action
                )


                action_domain = urlparse(
                    action_url
                ).netloc


                if (

                    action_domain

                    and action_domain != page_domain

                ):

                    external_form_count += 1


        results["external_form_count"] = (
            external_form_count
        )


        # ----------------------------------------------------
        # REDIRECT
        # ----------------------------------------------------

        results["redirect_count"] = len(
            redirect_chain
        )

        results["redirect_chain"] = (
            redirect_chain
        )


        # ----------------------------------------------------
        # LINKS
        # ----------------------------------------------------

        links = page.locator("a")

        link_count = links.count()

        external_link_count = 0

        same_domain_link_count = 0


        for i in range(link_count):

            link = links.nth(i)

            href = link.get_attribute(
                "href"
            )


            if not href:

                continue


            full_url = urljoin(
                page.url,
                href
            )


            link_domain = urlparse(
                full_url
            ).netloc


            if (

                link_domain

                and link_domain != page_domain

            ):

                external_link_count += 1


            elif link_domain == page_domain:

                same_domain_link_count += 1


        results["link_count"] = link_count

        results["external_link_count"] = (
            external_link_count
        )

        results["same_domain_link_count"] = (
            same_domain_link_count
        )


        # ----------------------------------------------------
        # IFRAMES
        # ----------------------------------------------------

        iframes = page.locator(
            "iframe"
        )

        iframe_count = iframes.count()

        external_iframe_count = 0


        for i in range(iframe_count):

            iframe = iframes.nth(i)

            src = iframe.get_attribute(
                "src"
            )


            if not src:

                continue


            iframe_url = urljoin(
                page.url,
                src
            )


            iframe_domain = urlparse(
                iframe_url
            ).netloc


            if (

                iframe_domain

                and iframe_domain != page_domain

            ):

                external_iframe_count += 1


        results["iframe_count"] = iframe_count

        results["external_iframe_count"] = (
            external_iframe_count
        )


        # ----------------------------------------------------
        # JAVASCRIPT
        # ----------------------------------------------------

        scripts = page.locator(
            "script"
        )

        script_count = scripts.count()

        external_script_count = 0

        inline_script_count = 0


        for i in range(script_count):

            script = scripts.nth(i)

            src = script.get_attribute(
                "src"
            )


            if src:

                external_script_count += 1

            else:

                inline_script_count += 1


        results["script_count"] = script_count

        results["external_script_count"] = (
            external_script_count
        )

        results["inline_script_count"] = (
            inline_script_count
        )


        # ----------------------------------------------------
        # IP ADDRESS
        # ----------------------------------------------------

        try:

            url_features = extract_features(
                page.url
            )


            results["is_domain_ip"] = int(
                url_features[
                    "IsDomainIP"
                ].iloc[0]
            )

        except:

            results["is_domain_ip"] = 0


        browser.close()


    return results


# ============================================================
# MAIN
# ============================================================

print(
    "\n========================================"
)

print(
    "       PHISHING WEBSITE SCANNER"
)

print(
    "========================================"
)


url = input(
    "\nEnter website URL: "
).strip()


if not url.startswith(
    ("http://", "https://")
):

    url = "https://" + url


# ============================================================
# ML
# ============================================================

print(
    "\nRunning Machine Learning analysis..."
)


ml_result, ml_confidence, features = (
    analyze_with_ml(url)
)


# ============================================================
# WEBSITE
# ============================================================

website = analyze_website(url)


# ============================================================
# RISK
# ============================================================

risk_score, risk_level, risk_reasons = (
    calculate_risk_score(
        ml_result,
        ml_confidence,
        website,
        features
    )
)


# ============================================================
# REPORT
# ============================================================

print(
    "\n========================================"
)

print(
    "             SCAN REPORT"
)

print(
    "========================================"
)


print(
    "\n---------- MACHINE LEARNING ----------"
)

print(
    "ML Prediction:",
    ml_result
)

print(
    "ML Confidence:",
    round(
        ml_confidence,
        2
    ),
    "%"
)

print(
    "---------------------------------------"
)


print(
    "\n---------- RISK ASSESSMENT ----------"
)

print(
    "PHISHING RISK SCORE:",
    risk_score,
    "/ 100"
)

print(
    "RISK LEVEL:",
    risk_level
)

print(
    "--------------------------------------"
)


print(
    "\n---------- DETECTION REASONS ----------"
)

for reason in risk_reasons:

    print(
        "✓",
        reason
    )

print(
    "----------------------------------------"
)


print(
    "\n---------- WEBSITE INFORMATION ----------"
)

print(
    "Final URL:",
    website["final_url"]
)

print(
    "Page Title:",
    website["title"]
)

print(
    "------------------------------------------"
)


print(
    "\n---------- SUSPICIOUS KEYWORDS ----------"
)

if website["suspicious_keywords"]:

    for keyword in website[
        "suspicious_keywords"
    ]:

        print(
            " -",
            keyword
        )

else:

    print(
        "No suspicious keywords detected."
    )

print(
    "------------------------------------------"
)


print(
    "\n---------- FORM ANALYSIS ----------"
)

print(
    "Forms:",
    website["form_count"]
)

print(
    "Input fields:",
    website["input_count"]
)

print(
    "Username/Email fields:",
    website["username_count"]
)

print(
    "Password fields:",
    website["password_count"]
)

print(
    "OTP fields:",
    website["otp_count"]
)

print(
    "External form destinations:",
    website["external_form_count"]
)

print(
    "----------------------------------"
)


print(
    "\n---------- REDIRECT ANALYSIS ----------"
)

print(
    "Redirects:",
    website["redirect_count"]
)

print(
    "----------------------------------------"
)


print(
    "\n---------- LINK ANALYSIS ----------"
)

print(
    "Total links:",
    website["link_count"]
)

print(
    "Same-domain links:",
    website[
        "same_domain_link_count"
    ]
)

print(
    "External links:",
    website[
        "external_link_count"
    ]
)

print(
    "----------------------------------"
)


print(
    "\n---------- IFRAME ANALYSIS ----------"
)

print(
    "Total iframes:",
    website["iframe_count"]
)

print(
    "External iframes:",
    website[
        "external_iframe_count"
    ]
)

print(
    "-------------------------------------"
)


print(
    "\n---------- JAVASCRIPT ANALYSIS ----------"
)

print(
    "Total scripts:",
    website["script_count"]
)

print(
    "External scripts:",
    website[
        "external_script_count"
    ]
)

print(
    "Inline scripts:",
    website[
        "inline_script_count"
    ]
)

print(
    "-----------------------------------------"
)


print(
    "\nScreenshot: website_screenshot.png"
)


print(
    "\n========================================"
)

print(
    "           END OF SCAN"
)

print(
    "========================================"
)