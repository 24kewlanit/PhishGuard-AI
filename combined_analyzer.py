"""
combined_analyzer.py
====================
STAGE 6 + 7 - COMBINE EMAIL + PDF RESULTS INTO ONE OVERALL VERDICT

Inputs (both come from other modules, each with a 0-100 risk score):
    email_result : {"classification": SAFE|SPAM|SUSPICIOUS|PHISHING|INCONCLUSIVE,        "risk_score", "reasons"}
    pdf_result   : {"classification": SAFE|SPAM|SUSPICIOUS|POTENTIALLY MALICIOUS|INCONCLUSIVE, "risk_score", "reasons"}
Either one may be None (e.g. the user only uploaded a PDF).
An INCONCLUSIVE part has risk_score None: it is "unknown", not "zero". It is
left out of the weighted score and it can NEVER let the overall result be SAFE.

STEP 1 - WEIGHTED SCORE
    overall = 0.60 * email_risk + 0.40 * pdf_risk
  Why 60/40? The email is the part the victim actually reads and acts on (the
  lure), and it is analysed with more signals (wording, sender, links). The PDF
  is the payload. Both matter, so neither weight is tiny. If only one input is
  present, it gets 100% of the weight.

STEP 2 - OVERRIDE FLOORS  (a weighted average alone is NOT trusted)
  An average lets a harmless-looking half hide a dangerous half, e.g.
  email 0 + PDF 90 -> average 36 ("looks fine"). So the final score can never
  fall below these floors:

    R1  Email is PHISHING                 -> final >= the email's own risk
    R2  PDF is POTENTIALLY MALICIOUS      -> final >= 55
        ... and PDF risk >= 85            -> final >= 70
        ... and email is not SAFE         -> final >= 70
    R3  PDF is SUSPICIOUS                 -> final >= 35
    R4  Email is SPAM  (or PDF is SPAM)   -> final >= 25
    R5  Email is SUSPICIOUS               -> final >= 35

STEP 3 - FINAL LABEL (checked in this order)
    final score >= 60                         PHISHING    (high-confidence malicious message)
    PDF is SUSPICIOUS / POTENTIALLY MALICIOUS SUSPICIOUS  (needs human review)
    email is SUSPICIOUS                       SUSPICIOUS
    email is SPAM or PDF is SPAM              SPAM
    final score >= 35                         SUSPICIOUS
    any part is INCONCLUSIVE                  INCONCLUSIVE (risk score not available)
    otherwise                                 SAFE

Reasons are copied from the modules that produced them. Nothing is invented;
the only extra lines are the explicit "override" explanations when a floor
actually raised the score.
"""

EMAIL_WEIGHT = 0.60
PDF_WEIGHT = 0.40

PHISHING_AT = 60
SUSPICIOUS_AT = 35


def combine_results(email_result=None, pdf_result=None):
    """Return {"classification", "risk_score", "reasons", "breakdown"}."""
    if email_result is None and pdf_result is None:
        raise ValueError("Nothing to combine: provide an email result, a PDF result, or both.")

    e_class = email_result["classification"] if email_result else None
    p_class = pdf_result["classification"] if pdf_result else None
    # An inconclusive part has no usable risk score ("unknown", not "zero").
    e_risk = email_result.get("risk_score") if email_result and e_class != "INCONCLUSIVE" else None
    p_risk = pdf_result.get("risk_score") if pdf_result and p_class != "INCONCLUSIVE" else None
    inconclusive = [name for name, cls in (("Email", e_class), ("PDF", p_class)) if cls == "INCONCLUSIVE"]

    # ---------- Step 1: weighted score (only parts with a real score) ----------
    if e_risk is not None and p_risk is not None:
        weights = {"email": EMAIL_WEIGHT, "pdf": PDF_WEIGHT}
        weighted = EMAIL_WEIGHT * e_risk + PDF_WEIGHT * p_risk
    elif e_risk is not None:
        weights = {"email": 1.0, "pdf": 0.0}
        weighted = float(e_risk)
    elif p_risk is not None:
        weights = {"email": 0.0, "pdf": 1.0}
        weighted = float(p_risk)
    else:
        weights = {"email": 0.0, "pdf": 0.0}
        weighted = None

    # ---------- Step 2: override floors ----------
    final = weighted
    override_reasons = []

    def raise_to(floor, why):
        nonlocal final
        if final is None or floor > final:
            final = floor
            override_reasons.append(why)

    if e_class == "PHISHING":
        raise_to(e_risk, "Override: the email itself is classified as phishing, so a safe-looking "
                         "attachment cannot lower the overall risk.")
    if p_class == "POTENTIALLY MALICIOUS":
        raise_to(55, "Override: the PDF shows strong malicious indicators, so the overall result "
                     "cannot be SAFE even if the email text looks harmless.")
        if p_risk >= 85:
            raise_to(70, "Override: the PDF risk is very high (85+).")
        if e_class not in (None, "SAFE", "INCONCLUSIVE"):
            raise_to(70, "Override: a potentially malicious PDF arrived in an email that is "
                         "itself not safe.")
    if p_class == "SUSPICIOUS":
        raise_to(35, "Override: the PDF contains suspicious content, so the overall "
                     "result cannot be SAFE.")
    if e_class == "SUSPICIOUS":
        raise_to(35, "Override: the email contains suspicious indicators, so the overall "
                     "result cannot be SAFE.")
    if e_class == "SPAM":
        raise_to(25, None)
    if p_class == "SPAM":
        raise_to(25, "Override: the PDF is spam / promotional content, so the overall result \
cannot be SAFE even if the email text looks harmless.")

    final = None if final is None else int(round(min(100, final)))

    # ---------- Step 3: label ----------
    if final is not None and final >= PHISHING_AT:
        label = "PHISHING"
    elif p_class in ("SUSPICIOUS", "POTENTIALLY MALICIOUS"):
        label = "SUSPICIOUS"          # active/dangerous PDF content is never just "spam"
    elif e_class == "SUSPICIOUS":
        label = "SUSPICIOUS"
    elif e_class == "SPAM" or p_class == "SPAM":
        label = "SPAM"                # spam with no dangerous attachment stays SPAM
    elif final is not None and final >= SUSPICIOUS_AT:
        label = "SUSPICIOUS"
    elif inconclusive:
        label = "INCONCLUSIVE"        # unknown is never SAFE
    else:
        label = "SAFE"

    if label == "INCONCLUSIVE":
        final = None                  # no honest risk number exists
    if inconclusive:
        override_reasons.append(
            "Override: the " + " and ".join(inconclusive) + " analysis was inconclusive, so the overall "
            "result cannot be SAFE" + ("." if label == "INCONCLUSIVE" else
                                      " (the overall verdict is based on the part that could be analysed).")
        )

    # ---------- reasons ----------
    reasons = [r for r in override_reasons if r]
    if email_result:
        reasons += [f"Email: {r}" for r in email_result.get("reasons", [])]
    if pdf_result:
        reasons += [f"PDF: {r}" for r in pdf_result.get("reasons", [])]
    if not reasons:
        reasons = ["No suspicious indicators were detected by the checks that were run "
                   "(this is not a guarantee that the content is safe)."]

    return {
        "classification": label,
        "risk_score": final,
        "reasons": reasons,
        "breakdown": {
            "email_risk": e_risk,
            "pdf_risk": p_risk,
            "weights": weights,
            "weighted_score": None if weighted is None else round(weighted, 1),
            "email_classification": e_class,
            "pdf_classification": p_class,
        },
    }
