# Email detection fix

## What changed
- Added a targeted, explainable signal for unsolicited internship/recruitment campaigns.
- Added a second signal when recruitment copy also uses application pressure or benefit claims (for example, “no interviews”, “limited seats”, “register now”, “offer letter”, or “completion certificate”).
- These signals classify the message as **SPAM**, not automatically as confirmed phishing. The scanner still reports evidence and risk separately.
- Added `test_email_regression.py` to prevent the two reported false-safe cases from returning unnoticed.

## Run the regression test
From this project folder:

```bash
python test_email_regression.py
```

## Important limitation
No email detector can guarantee perfect classification for every email worldwide. This project’s email model was trained on synthetic/template-generated examples, not a representative global corpus of real emails. The score is not a calibrated probability. Sender authentication (SPF/DKIM/DMARC) can pass for spam and does not prove that an offer is trustworthy. For stronger real-world performance, train and validate on a large, diverse, properly labeled dataset containing safe mail, bulk spam, phishing, business-email compromise, job/internship scams, multiple languages, and current attack patterns. Evaluate on a separate time-based and sender/domain-separated test set.
