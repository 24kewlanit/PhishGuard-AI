"""Regression checks for recruitment-spam emails. Run: python test_email_regression.py"""
from email_analyzer import parse_email_text
from email_model import load_email_model, predict_email

MODEL = load_email_model()

CASES = [
    {
        "name": "internship openings campaign",
        "text": """From: Internship <snehajha1412@gmail.com>
Subject: Internship openings
Date: Thu, 1 Oct 2026 05:13:43 +0000

Internship Program Details. Duration: 1 Month. Last Date to Register: 3rd October.
Work from Home. Perks: Offer Letter & Completion Certificate.
Internship Application Link: https://example.com/internships/""",
        "expected": "SPAM",
    },
    {
        "name": "no-interview virtual internship campaign",
        "text": """From: Internship <learntricks1002@gmail.com>
Subject: Internship Program 2026
Date: Sat, 12 Sep 2026 00:57:07 -0700

We invite you to our Internship Program. Virtual Internship.
No interviews, just learning and growth! Limited seats available.
Register now for an offer letter and completion certificate:
https://example.org/internships/""",
        "expected": "SPAM",
    },
]

def main():
    failed = 0
    for case in CASES:
        parsed = parse_email_text(case["text"])
        result = predict_email(parsed, MODEL)
        passed = result["classification"] == case["expected"]
        print(f'{"PASS" if passed else "FAIL"} | {case["name"]}: '
              f'{result["classification"]} (risk={result["risk_score"]})')
        if not passed:
            failed += 1
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
