/* =========================================================
   PHISHGUARD AI - FINAL SCRIPT
========================================================= */

"use strict";


/* =========================================================
   DOM READY
========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const urlInput = document.getElementById("urlInput");

    if (urlInput) {

        urlInput.addEventListener("keydown", function (event) {

            if (event.key === "Enter") {

                event.preventDefault();

                scanURL();

            }

        });

    }

    loadHistory();

});


/* =========================================================
   SCAN URL
========================================================= */

async function scanURL() {

    const urlInput = document.getElementById("urlInput");
    const scanButton = document.getElementById("scanButton");
    const buttonText = document.getElementById("buttonText");
    const errorMessage = document.getElementById("errorMessage");

    const scanning = document.getElementById("scanning");
    const resultSection = document.getElementById("resultSection");

    const progressBar = document.getElementById("progressBar");
    const progressText = document.getElementById("progressText");
    const scanningText = document.getElementById("scanningText");

    if (!urlInput) {
        console.error("urlInput not found");
        return;
    }

    const url = urlInput.value.trim();

    errorMessage.textContent = "";

    resultSection.classList.add("hidden");

    if (!url) {

        errorMessage.textContent =
            "Please enter a website URL.";

        return;
    }


    /* -----------------------------------------
       BASIC URL VALIDATION
    ----------------------------------------- */

    let testURL = url;

    if (
        !testURL.startsWith("http://") &&
        !testURL.startsWith("https://")
    ) {
        testURL = "https://" + testURL;
    }

    try {

        const parsed = new URL(testURL);

        if (!parsed.hostname) {
            throw new Error();
        }

    } catch {

        errorMessage.textContent =
            "Please enter a valid website URL.";

        return;
    }


    /* -----------------------------------------
       START SCANNING
    ----------------------------------------- */

    scanButton.disabled = true;

    buttonText.textContent =
        "SCANNING...";

    scanning.classList.remove("hidden");


    let progress = 0;

    progressBar.style.width = "0%";

    progressText.textContent = "0%";

    scanningText.textContent =
        "Starting security analysis...";


    const messages = [

        "Extracting URL features...",

        "Analyzing domain structure...",

        "Checking suspicious patterns...",

        "Inspecting website behavior...",

        "Running machine learning model...",

        "Calculating threat score..."

    ];


    let messageIndex = 0;


    const progressInterval = setInterval(function () {

        progress += 2;

        if (progress > 92) {
            progress = 92;
        }

        progressBar.style.width =
            progress + "%";

        progressText.textContent =
            progress + "%";


        if (
            progress % 14 === 0 &&
            messageIndex < messages.length
        ) {

            scanningText.textContent =
                messages[messageIndex];

            messageIndex++;

        }

    }, 100);


    /* -----------------------------------------
       SEND REQUEST
    ----------------------------------------- */

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },

                body: JSON.stringify({
                    url: testURL
                })
            }
        );


        if (!response.ok) {

            throw new Error(
                "Server returned HTTP " +
                response.status
            );

        }


        const data = await response.json();


        clearInterval(progressInterval);


        /* -------------------------------------
           COMPLETE PROGRESS
        ------------------------------------- */

        progressBar.style.width = "100%";

        progressText.textContent = "100%";

        scanningText.textContent =
            "Analysis complete.";


        await new Promise(resolve =>
            setTimeout(resolve, 350)
        );


        scanning.classList.add("hidden");


        /* -------------------------------------
           SERVER ERROR
        ------------------------------------- */

        if (!data.success) {

            errorMessage.textContent =
                data.error ||
                "Unable to analyze this URL.";

            scanButton.disabled = false;

            buttonText.textContent =
                "SCAN URL";

            return;
        }


        /* -------------------------------------
           DISPLAY RESULT
        ------------------------------------- */

        displayResult(data);

        loadHistory();


        scanButton.disabled = false;

        buttonText.textContent =
            "SCAN URL";


    } catch (error) {

        clearInterval(progressInterval);

        scanning.classList.add("hidden");


        console.error(
            "SCAN ERROR:",
            error
        );


        errorMessage.textContent =
            "Unable to connect to the detection system. " +
            "Make sure Flask is running.";


        scanButton.disabled = false;

        buttonText.textContent =
            "SCAN URL";
    }
}


/* =========================================================
   DISPLAY RESULT
========================================================= */

function displayResult(data) {

    const resultSection =
        document.getElementById("resultSection");

    const resultCard =
        document.getElementById("resultCard");

    const resultIcon =
        document.getElementById("resultIcon");

    const resultLabel =
        document.getElementById("resultLabel");

    const resultDescription =
        document.getElementById("resultDescription");

    const resultBadge =
        document.getElementById("resultBadge");

    const scannedURL =
        document.getElementById("scannedURL");


    /* -----------------------------------------
       PROBABILITY
    ----------------------------------------- */

    const riskPercentage =
        document.getElementById("riskPercentage");

    const phishingProbability =
        document.getElementById("phishingProbability");

    const legitimateProbability =
        document.getElementById("legitimateProbability");

    const phishingBar =
        document.getElementById("phishingBar");

    const legitimateBar =
        document.getElementById("legitimateBar");

    const riskCircle =
        document.getElementById("riskCircle");


    /* -----------------------------------------
       FEATURES
    ----------------------------------------- */

    const featuresGrid =
        document.getElementById("featuresGrid");

    const featureCount =
        document.getElementById("featureCount");


    /* -----------------------------------------
       WEBSITE
    ----------------------------------------- */

    const pageTitle =
        document.getElementById("pageTitle");

    const formCount =
        document.getElementById("formCount");

    const inputCount =
        document.getElementById("inputCount");

    const usernameCount =
        document.getElementById("usernameCount");

    const passwordCount =
        document.getElementById("passwordCount");

    const otpCount =
        document.getElementById("otpCount");

    const externalFormCount =
        document.getElementById("externalFormCount");

    const redirectCount =
        document.getElementById("redirectCount");

    const linkCount =
        document.getElementById("linkCount");

    const externalLinkCount =
        document.getElementById("externalLinkCount");

    const externalIframeCount =
        document.getElementById("externalIframeCount");

    const externalScriptCount =
        document.getElementById("externalScriptCount");


    /* -----------------------------------------
       OTHER
    ----------------------------------------- */

    const suspiciousKeywords =
        document.getElementById("suspiciousKeywords");

    const riskReasons =
        document.getElementById("riskReasons");

    const loadErrorNotice =
        document.getElementById("loadErrorNotice");

    const screenshotSection =
        document.getElementById("screenshotSection");

    const screenshotImage =
        document.getElementById("screenshotImage");

    const redirectSection =
        document.getElementById("redirectSection");

    const redirectChain =
        document.getElementById("redirectChain");


    /* =====================================================
       SHOW RESULT SECTION
    ===================================================== */

    resultSection.classList.remove("hidden");


    /* =====================================================
       URL
    ===================================================== */

    scannedURL.textContent =
        data.final_url || data.url || "Unknown URL";


    /* =====================================================
       ML PROBABILITIES
    ===================================================== */

    const phishing =
        Number(data.phishing_probability) || 0;

    const legitimate =
        Number(data.legitimate_probability) || 0;


    phishingProbability.textContent =
        phishing.toFixed(2) + "%";

    legitimateProbability.textContent =
        legitimate.toFixed(2) + "%";


    phishingBar.style.width =
        Math.min(Math.max(phishing, 0), 100) + "%";

    legitimateBar.style.width =
        Math.min(Math.max(legitimate, 0), 100) + "%";


    /* =====================================================
       RISK SCORE
    ===================================================== */

    const riskScore =
        Number(data.risk_score) || 0;


    riskPercentage.textContent =
        Math.round(riskScore) + "%";


    riskCircle.style.setProperty(
        "--risk",
        riskScore
    );


    /* =====================================================
       RESULT STATUS
    ===================================================== */

    resultCard.classList.remove(
        "phishing",
        "legitimate",
        "suspicious"
    );


    if (data.result === "PHISHING") {

        resultCard.classList.add(
            "phishing"
        );

        resultIcon.textContent =
            "⚠";

        resultLabel.textContent =
            "PHISHING DETECTED";

        resultDescription.textContent =
            "Warning: This URL may be dangerous or suspicious.";

        resultBadge.textContent =
            "THREAT";

    } else if (data.result === "SUSPICIOUS") {

        resultCard.classList.add(
            "suspicious"
        );

        resultIcon.textContent =
            "⚠";

        resultLabel.textContent =
            "SUSPICIOUS";

        resultDescription.textContent =
            "This URL shows some risky signals. Proceed with caution.";

        resultBadge.textContent =
            "CAUTION";

    } else {

        resultCard.classList.add(
            "legitimate"
        );

        resultIcon.textContent =
            "✓";

        resultLabel.textContent =
            "LEGITIMATE";

        resultDescription.textContent =
            "This URL appears to be legitimate.";

        resultBadge.textContent =
            "SAFE";
    }

    // Make the decision layer explicit when a trusted exact domain is recognized.
    if (data.domain_trusted) {
        resultDescription.textContent =
            "This domain matches a configured trusted domain. The ML score is shown below as a supporting signal.";
    }


    /* =====================================================
       URL FEATURES
    ===================================================== */

    featuresGrid.innerHTML = "";


    const featureNames = {
        URLLength: "URL Length",
        DomainLength: "Domain Length",
        IsDomainIP: "Domain is IP",
        NoOfSubDomain: "Subdomains",
        NoOfLettersInURL: "Letters",
        NoOfDegitsInURL: "Digits",
        NoOfEqualsInURL: "Equals (=)",
        NoOfQMarkInURL: "Question Marks (?)",
        NoOfAmpersandInURL: "Ampersands (&)",
        NoOfOtherSpecialCharsInURL: "Other Special Characters",
        IsHTTPS: "HTTPS",
        TLDLength: "TLD Length",
        NoOfObfuscatedChar: "Encoded Characters",
        ObfuscationRatio: "Obfuscation Ratio",
        LetterRatioInURL: "Letter Ratio",
        DegitRatioInURL: "Digit Ratio",
        SpacialCharRatioInURL: "Special Character Ratio"
    };


    if (data.features) {

        Object.entries(data.features).forEach(
            function ([key, value]) {

                const feature =
                    document.createElement("div");

                feature.className =
                    "feature-item";


                let displayValue = value;


                if (
                    key === "LetterRatioInURL" ||
                    key === "DegitRatioInURL" ||
                    key === "SpacialCharRatioInURL" ||
                    
                    key === "Entropy"
                ) {

                    displayValue =
                        Number(value).toFixed(4);

                }


                const name =
                    featureNames[key] || key;


                const nameDiv =
                    document.createElement("div");

                nameDiv.className =
                    "feature-name";

                nameDiv.textContent =
                    name;


                const valueDiv =
                    document.createElement("div");

                valueDiv.className =
                    "feature-value";

                valueDiv.textContent =
                    displayValue;


                feature.appendChild(
                    nameDiv
                );

                feature.appendChild(
                    valueDiv
                );


                featuresGrid.appendChild(
                    feature
                );

            }
        );


        featureCount.textContent =
            Object.keys(data.features).length +
            " FEATURES";
    }


    /* =====================================================
       WEBSITE ANALYSIS
    ===================================================== */

    pageTitle.textContent =
        data.title || "Page unavailable";

    formCount.textContent =
        data.form_count ?? 0;

    inputCount.textContent =
        data.input_count ?? 0;

    usernameCount.textContent =
        data.username_count ?? 0;

    passwordCount.textContent =
        data.password_count ?? 0;

    otpCount.textContent =
        data.otp_count ?? 0;

    externalFormCount.textContent =
        data.external_form_count ?? 0;

    redirectCount.textContent =
        data.redirect_count ?? 0;

    linkCount.textContent =
        data.link_count ?? 0;

    externalLinkCount.textContent =
        data.external_link_count ?? 0;

    externalIframeCount.textContent =
        data.external_iframe_count ?? 0;

    externalScriptCount.textContent =
        data.external_script_count ?? 0;


    /* =====================================================
       SECURITY TERMS
    ===================================================== */

    suspiciousKeywords.innerHTML = "";

    const keywords =
        Array.isArray(data.suspicious_keywords)
            ? data.suspicious_keywords
            : [];

    const termHeader = document.createElement("div");
    termHeader.className = "terms-note";

    if (data.domain_trusted && keywords.length > 0) {
        termHeader.innerHTML =
            "<strong>Contextual terms</strong><span>These terms are common on this trusted website and are not treated as phishing evidence by themselves.</span>";
    } else if (keywords.length > 0) {
        termHeader.innerHTML =
            "<strong>Detected terms</strong><span>These words are shown as context. They are not proof of phishing on their own.</span>";
    } else {
        termHeader.innerHTML =
            "<strong>No notable security terms detected</strong><span>No configured credential or payment-related terms were found in the page text.</span>";
    }

    suspiciousKeywords.appendChild(termHeader);

    if (keywords.length > 0) {
        const list = document.createElement("div");
        list.className = "keyword-list-inner";

        keywords.forEach(function (keyword) {
            const item = document.createElement("span");
            item.className = "keyword";
            item.textContent = keyword;
            list.appendChild(item);
        });

        suspiciousKeywords.appendChild(list);
    }


    /* =====================================================
       LOAD ERROR
    ===================================================== */

    if (data.load_error) {

        loadErrorNotice.textContent =
            "⚠ " + data.load_error;

        loadErrorNotice.classList.remove(
            "hidden"
        );

    } else {

        loadErrorNotice.classList.add(
            "hidden"
        );
    }


    /* =====================================================
       SCREENSHOT
    ===================================================== */

    if (data.screenshot_url) {

        screenshotImage.src =
            data.screenshot_url +
            "?t=" +
            Date.now();

        screenshotSection.classList.remove(
            "hidden"
        );

    } else {

        screenshotImage.src = "";

        screenshotSection.classList.add(
            "hidden"
        );
    }


    /* =====================================================
       REDIRECT CHAIN
    ===================================================== */

    redirectChain.innerHTML = "";


    const redirects =
        Array.isArray(data.redirect_chain)
            ? data.redirect_chain
            : [];


    if (redirects.length > 0) {
        redirects.forEach(function (step, index) {
            const item = document.createElement("div");
            item.className = "redirect-step";

            const marker = document.createElement("div");
            marker.className = "redirect-marker";
            marker.textContent = index + 1;

            const body = document.createElement("div");
            body.className = "redirect-body";

            const top = document.createElement("div");
            top.className = "redirect-top";

            const raw = String(step || "");
            const match = raw.match(/^(\d{3})\s*[-→>]+\s*(.*)$/);
            const code = match ? match[1] : "INFO";
            const target = match ? match[2] : raw;

            const badge = document.createElement("span");
            badge.className = "redirect-code " + (code.startsWith("2") ? "ok" : code.startsWith("3") ? "redirect" : "warn");
            badge.textContent = code;

            const label = document.createElement("span");
            label.className = "redirect-label";
            label.textContent = index === 0 ? "Initial redirect" : "Redirect hop";

            top.appendChild(badge);
            top.appendChild(label);

            const url = document.createElement("div");
            url.className = "redirect-url";
            url.textContent = target;

            body.appendChild(top);
            body.appendChild(url);
            item.appendChild(marker);
            item.appendChild(body);
            redirectChain.appendChild(item);
        });

        redirectSection.classList.remove("hidden");
    } else {
        redirectSection.classList.remove("hidden");
        redirectChain.innerHTML = '<div class="redirect-none"><span>✓</span><div><strong>No redirect hops detected</strong><small>The final page was reached without a recorded redirect chain.</small></div></div>';
    }


    /* =====================================================
       DETECTION REASONS
    ===================================================== */

    riskReasons.innerHTML = "";


    const reasons =
        Array.isArray(data.risk_reasons)
            ? data.risk_reasons
            : [];


    if (reasons.length > 0) {

        reasons.forEach(
            function (reason) {

                const item =
                    document.createElement("div");

                item.className =
                    "reason-item";


                const icon =
                    document.createElement("span");

                icon.textContent =
                    "✓";


                const text =
                    document.createElement("span");

                text.textContent =
                    reason;


                item.appendChild(icon);

                item.appendChild(text);


                riskReasons.appendChild(
                    item
                );

            }
        );

    } else {

        riskReasons.textContent =
            "No major suspicious indicators detected.";
    }


    /* =====================================================
       SCROLL
    ===================================================== */

    setTimeout(function () {

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }, 200);
}


/* =========================================================
   RESET SCANNER
========================================================= */

function resetScanner() {

    const urlInput =
        document.getElementById("urlInput");

    const resultSection =
        document.getElementById("resultSection");

    const scanning =
        document.getElementById("scanning");

    const errorMessage =
        document.getElementById("errorMessage");

    const progressBar =
        document.getElementById("progressBar");

    const progressText =
        document.getElementById("progressText");

    const scanningText =
        document.getElementById("scanningText");

    const scanButton =
        document.getElementById("scanButton");

    const buttonText =
        document.getElementById("buttonText");


    urlInput.value = "";

    resultSection.classList.add(
        "hidden"
    );

    scanning.classList.add(
        "hidden"
    );

    errorMessage.textContent = "";


    progressBar.style.width =
        "0%";

    progressText.textContent =
        "0%";

    scanningText.textContent =
        "Starting security analysis...";


    scanButton.disabled = false;

    buttonText.textContent =
        "SCAN URL";


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });


    setTimeout(function () {

        urlInput.focus();

    }, 300);
}


/* =========================================================
   HISTORY
========================================================= */

async function loadHistory() {

    try {

        const response =
            await fetch("/history");


        if (!response.ok) {
            throw new Error(
                "History request failed"
            );
        }


        const data =
            await response.json();


        if (data.success) {

            renderHistory(
                data.history || []
            );

        }

    } catch (error) {

        console.error(
            "Unable to load history:",
            error
        );
    }
}


/* =========================================================
   RENDER HISTORY
========================================================= */

function renderHistory(items) {

    const historyList =
        document.getElementById(
            "historyList"
        );


    if (!historyList) {
        return;
    }


    historyList.innerHTML = "";


    if (
        !Array.isArray(items) ||
        items.length === 0
    ) {

        const empty =
            document.createElement("div");

        empty.className =
            "history-empty";

        empty.textContent =
            "No scans yet. Run your first scan above.";

        historyList.appendChild(
            empty
        );

        return;
    }


    items.forEach(
        function (item) {

            const row =
                document.createElement("div");

            row.className =
                "history-item";

            row.title =
                "Click to scan this URL again";


            const urlDiv =
                document.createElement("div");

            urlDiv.className =
                "history-item-url";

            urlDiv.textContent =
                item.url;


            const meta =
                document.createElement("div");

            meta.className =
                "history-item-meta";


            const time =
                document.createElement("span");

            time.className =
                "history-item-time";

            time.textContent =
                item.timestamp;


            const badge =
                document.createElement("span");

            let badgeClass = "legitimate";
            let badgeLabel = "SAFE";

            if (item.result === "PHISHING") {
                badgeClass = "phishing";
                badgeLabel = "PHISHING";
            } else if (item.result === "SUSPICIOUS") {
                badgeClass = "suspicious";
                badgeLabel = "SUSPICIOUS";
            }


            badge.className =
                "history-badge " +
                badgeClass;


            badge.textContent =
                badgeLabel +
                " · " +
                (item.risk_score ?? 0) +
                "%";


            meta.appendChild(time);

            meta.appendChild(badge);


            row.appendChild(urlDiv);

            row.appendChild(meta);


            row.addEventListener(
                "click",
                function () {

                    const input =
                        document.getElementById(
                            "urlInput"
                        );

                    input.value =
                        item.url;

                    scanURL();

                }
            );


            historyList.appendChild(
                row
            );

        }
    );
}